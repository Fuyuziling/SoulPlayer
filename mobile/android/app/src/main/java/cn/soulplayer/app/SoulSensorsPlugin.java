package cn.soulplayer.app;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.view.WindowManager;

import androidx.core.content.ContextCompat;

import com.getcapacitor.JSArray;
import com.getcapacitor.JSObject;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;
import com.getcapacitor.annotation.Permission;
import com.getcapacitor.annotation.PermissionCallback;

/**
 * 原生 IMU 采集插件。
 *
 * 为什么不直接用 WebView 的 devicemotion：
 *   ① 可用性取决于系统 WebView 版本，部分机型根本不触发，且不报错；
 *   ② 事件被合并到渲染节拍，通常封顶 60Hz，而甩动全程只有 0.2~0.3s，
 *      十几帧算不准跨度与峰值速度；
 *   ③ rotationRate 的 alpha/beta/gamma 轴向约定在各浏览器实现里不一致，
 *      而判定链的坐标约定是固定的（屏幕竖直、x 右 y 上 z 朝外）。
 *
 * 这里直接读 SensorManager，保持与 TS 侧 core/types.ts 完全一致的坐标语义，
 * 成批（40ms）跨桥推送，压住 JSON 开销。判定逻辑一行都不改。
 */
@CapacitorPlugin(
        name = "SoulSensors",
        permissions = {
            @Permission(alias = "activity", strings = { Manifest.permission.ACTIVITY_RECOGNITION })
        })
public class SoulSensorsPlugin extends Plugin implements SensorEventListener {

    /** 100Hz。甩动全程 0.2~0.3s，50Hz 只有十几帧，特征会明显失真。 */
    private static final int SAMPLE_PERIOD_US = 10_000;

    /**
     * 40ms 推一批。跨桥一次约 4 帧，既压住消息量，又不至于让延迟影响手感
     * （判定窗口最小 360ms，40ms 的批次延迟在噪声范围内）。
     */
    private static final long FLUSH_INTERVAL_MS = 40;

    private SensorManager sensorManager;
    private Sensor accelerometer;
    private Sensor gyroscope;
    /** 硬件步检测器（TYPE_STEP_DETECTOR）。可能为 null：部分低端机型没有。 */
    private Sensor stepDetector;

    private final Handler handler = new Handler(Looper.getMainLooper());
    private JSArray pending = new JSArray();
    private final Object lock = new Object();

    private boolean running = false;
    private float[] gyroValues = null;
    /** 步监听当前是否已注册。退后台注销、回前台重挂，跟着生命周期走。 */
    private boolean stepListening = false;
    /**
     * 步检测是否已开通（权限到手过）。与 stepListening 分开：
     * unregisterListeners 会清掉后者，回前台要凭前者把监听挂回来——
     * 共用一个标志的话，退一次后台步检测就永久失效了。
     */
    private boolean stepEnabled = false;

    private final Runnable flushTask = new Runnable() {
        @Override
        public void run() {
            if (!running) return;
            flush();
            handler.postDelayed(this, FLUSH_INTERVAL_MS);
        }
    };

    /** 懒初始化：getContext() 要等 bridge 建好才可用。 */
    private void ensureSensors() {
        if (sensorManager != null) return;
        Context ctx = getContext();
        sensorManager = (SensorManager) ctx.getSystemService(Context.SENSOR_SERVICE);
        if (sensorManager == null) return;
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        stepDetector = sensorManager.getDefaultSensor(Sensor.TYPE_STEP_DETECTOR);
    }

    /**
     * 步检测在 Android 10+ 需要 ACTIVITY_RECOGNITION 运行时权限。
     * 没有权限时传感器拿得到但不出事件——静默失效比崩溃更难查，
     * 所以这里显式查权限，未授权就当作"不可用"，JS 侧保持中性降级。
     */
    private boolean hasActivityPermission() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) return true;
        return ContextCompat.checkSelfPermission(
                getContext(), Manifest.permission.ACTIVITY_RECOGNITION)
                == PackageManager.PERMISSION_GRANTED;
    }

    private void notifyStepDetector(boolean available) {
        JSObject ret = new JSObject();
        ret.put("available", available);
        notifyListeners("stepdetector", ret);
    }

    private JSObject probe(boolean available, String message) {
        JSObject ret = new JSObject();
        ret.put("available", available);
        ret.put("hasGyro", gyroscope != null);
        ret.put("hasStepDetector", stepDetector != null && hasActivityPermission());
        if (message != null) ret.put("message", message);
        return ret;
    }

    @PluginMethod
    public void isAvailable(PluginCall call) {
        ensureSensors();
        if (sensorManager == null || accelerometer == null) {
            call.resolve(probe(false, "这台设备没有加速度计，无法做动作判定"));
            return;
        }
        call.resolve(probe(true, null));
    }

    @PluginMethod
    public void start(PluginCall call) {
        ensureSensors();
        if (sensorManager == null || accelerometer == null) {
            call.resolve(probe(false, "这台设备没有加速度计，无法做动作判定"));
            return;
        }
        if (!running) {
            registerListeners();
            running = true;
            handler.postDelayed(flushTask, FLUSH_INTERVAL_MS);
            maybeStartStepDetector();
        }
        keepScreenOn(true);
        call.resolve(probe(true, null));
    }

    @PluginMethod
    public void stop(PluginCall call) {
        if (running) {
            running = false;
            handler.removeCallbacks(flushTask);
            flush();               // 把最后一批样本送出去，别丢尾巴
            unregisterListeners();
        }
        keepScreenOn(false);
        call.resolve();
    }

    private void registerListeners() {
        sensorManager.registerListener(this, accelerometer, SAMPLE_PERIOD_US);
        if (gyroscope != null) {
            sensorManager.registerListener(this, gyroscope, SAMPLE_PERIOD_US);
        }
    }

    private void unregisterListeners() {
        if (sensorManager != null) sensorManager.unregisterListener(this);
        gyroValues = null;
        // 只清"已注册"，不清"已开通"——回前台凭 stepEnabled 重挂步监听。
        stepListening = false;
    }

    /**
     * 步检测器：有传感器且权限已到手就直接开（重启 App 后权限仍在，这条是常态路径）。
     * 权限未到手时**不在这里弹窗**——start() 是 JS 主动调用的第一个方法，
     * 此时 UI 还在校准引导里，突然弹系统权限框体验很差。申请交给 JS 在
     * 合适时机调 requestStepPermission()。
     */
    private void maybeStartStepDetector() {
        if (stepDetector == null) return;
        if (hasActivityPermission()) registerStepListener();
    }

    /**
     * JS 主动申请步检测权限（Android 10+ 需要 ACTIVITY_RECOGNITION）。
     * 走 Capacitor 官方 PermissionCallback 通道而不是裸 ActivityCompat——
     * 裸请求的结果回不到插件，授权了也收不到通知。
     * 拒绝也没关系：JS 按"不可用"中性降级，判定链行为退回旧版。
     */
    @PluginMethod
    public void requestStepPermission(PluginCall call) {
        ensureSensors();
        if (stepDetector == null) {
            JSObject ret = new JSObject();
            ret.put("granted", false);
            ret.put("reason", "no-sensor");
            call.resolve(ret);
            return;
        }
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q || hasActivityPermission()) {
            // 老系统不需要运行时权限；已授权的直接开，不弹窗。
            registerStepListener();
            JSObject ret = new JSObject();
            ret.put("granted", true);
            call.resolve(ret);
            return;
        }
        requestPermissionForAlias("activity", call, "stepPermissionResult");
    }

    @PermissionCallback
    private void stepPermissionResult(PluginCall call) {
        boolean granted = hasActivityPermission();
        if (granted) {
            registerStepListener();
        } else {
            notifyStepDetector(false);
        }
        JSObject ret = new JSObject();
        ret.put("granted", granted);
        call.resolve(ret);
    }

    private void registerStepListener() {
        if (stepDetector == null || stepListening) return;
        // 用户可能在系统设置里收回过权限，重挂前再查一次：
        // 没权限的注册拿得到传感器但永远不出事件，等于静默失效。
        if (!hasActivityPermission()) return;
        // 采样周期参数对步检测器无效（它由步事件驱动），给 NORMAL 只是占位。
        // 这是硬件级低功耗传感器，注册着几乎不耗电。
        sensorManager.registerListener(this, stepDetector, SensorManager.SENSOR_DELAY_NORMAL);
        stepEnabled = true;
        stepListening = true;
        notifyStepDetector(true);
    }

    /**
     * 跟着 Activity 生命周期走：退到后台就停采，回来再续上。
     * JS 侧不感知这件事——它只知道"通道是开着的"，样本会自己续上来。
     * 屏幕常亮只在采集期间保持，否则放口袋里也亮屏，纯耗电。
     */
    @Override
    protected void handleOnPause() {
        if (running) {
            handler.removeCallbacks(flushTask);
            unregisterListeners();
        }
        super.handleOnPause();
    }

    @Override
    protected void handleOnResume() {
        super.handleOnResume();
        if (running) {
            registerListeners();
            // 旧代码判断 stepListening，但退后台时它已被 unregisterListeners 清零，
            // 这条分支永远走不到——回前台后步检测静默失效。改凭 stepEnabled 判断。
            if (stepEnabled) registerStepListener();
            handler.removeCallbacks(flushTask);
            handler.postDelayed(flushTask, FLUSH_INTERVAL_MS);
        }
    }

    private void keepScreenOn(boolean on) {
        if (getActivity() == null) return;
        getActivity().runOnUiThread(() -> {
            if (on) {
                getActivity().getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
            } else {
                getActivity().getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
            }
        });
    }

    // ── SensorEventListener ─────────────────────────────

    @Override
    public void onSensorChanged(SensorEvent event) {
        if (event.sensor.getType() == Sensor.TYPE_GYROSCOPE) {
            // 陀螺仪只记"最新值"，配对到下一帧加速度上。两者都在 100Hz，
            // 时间偏差不超过一个采样周期（10ms），对判定无实质影响。
            gyroValues = event.values.clone();
            return;
        }
        if (event.sensor.getType() == Sensor.TYPE_STEP_DETECTOR) {
            // 硬件判出一步（脚触地时刻）。频率不超过 ~5Hz，逐条直发即可，
            // 不值得进 40ms 批量通道。post 到主线程再跨桥，与 flush 同一条路。
            final double t = round6(event.timestamp / 1e9);
            handler.post(() -> {
                JSObject ret = new JSObject();
                ret.put("t", t);
                notifyListeners("step", ret);
            });
            return;
        }
        if (event.sensor.getType() != Sensor.TYPE_ACCELEROMETER) return;

        float[] g = gyroValues;
        boolean hasRot = g != null;

        JSObject s = new JSObject();
        // 系统时钟（开机纳秒）转秒。单调、不受系统时间调整影响，
        // 与 JS 侧 performance.now() 不在同一基准，但判定链只用时间差，无碍。
        s.put("t", round6(event.timestamp / 1e9));
        s.put("ax", round4(event.values[0]));
        s.put("ay", round4(event.values[1]));
        s.put("az", round4(event.values[2]));
        s.put("gx", hasRot ? round4(g[0]) : 0);
        s.put("gy", hasRot ? round4(g[1]) : 0);
        s.put("gz", hasRot ? round4(g[2]) : 0);
        s.put("rot", hasRot);

        synchronized (lock) {
            pending.put(s);
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
        // 不做处理：判定链自己维护噪声底（校准流程会实测），
        // 系统的 accuracy 档位在不同厂商 ROM 上含义不一致，信不过。
    }

    private void flush() {
        JSArray batch;
        synchronized (lock) {
            if (pending.length() == 0) return;
            batch = pending;
            pending = new JSArray();
        }
        JSObject ret = new JSObject();
        ret.put("samples", batch);
        notifyListeners("samples", ret);
    }

    /** 跨桥 JSON 里保留 4 位小数足够：0.0001 m/s² 远小于噪声底。 */
    private static double round4(float v) {
        return Math.round(v * 10000.0) / 10000.0;
    }

    private static double round6(double v) {
        return Math.round(v * 1_000_000.0) / 1_000_000.0;
    }
}
