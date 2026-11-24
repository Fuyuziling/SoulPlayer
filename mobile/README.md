# SoulPlayer 手机端 / Android 安装包

手机端用 **Vite + TypeScript** 写判定链与界面，用 **Capacitor** 套一层原生壳打成 APK。
同一份 `src/` 既产出 PWA（浏览器直开），也产出 APK（传感器走原生通道），不维护两套代码。

## 两条运行通道

| | PWA（浏览器） | APK（Android） |
|---|---|---|
| 采集 | `devicemotion`（`platform/sensors.ts`） | `SensorManager`（`platform/native-sensors.ts` + `SoulSensorsPlugin.java`） |
| 采样率 | 受渲染节拍限制，一般 ≤60Hz | 100Hz |
| 安装 | 加主屏幕 | 侧载 APK |
| 选择逻辑 | `platform/motion.ts` 按 `Capacitor.isNativePlatform()` 决定 | 同 |

判定链 `src/core/` 完全不感知通道差异：两条通道都只负责产出
`{t, ax, ay, az, gx, gy, gz, hasRotation}`。

**为什么 APK 内不用 devicemotion**：WebView 对它的支持取决于系统 WebView 版本，
部分机型根本不触发且不报错；事件还会被合并到渲染节拍，60Hz 对 0.2~0.3s 的甩动来说
只有十几帧，跨度与峰值速度都算不准；`rotationRate` 的轴向约定在各实现里也不一致。
原生直读传感器后这三点全部消失。

## 构建

工具链位置通过 `ANDROID_HOME` / `JAVA_HOME` 环境变量注入，脚本内置一份可改的默认路径
（见 `tools/build_apk.py` 顶部常量，按自己机器调整即可），不需要改动系统环境变量。

| 组件 | 要求 | 说明 |
|---|---|---|
| JDK | 17 | Capacitor 6 + AGP 8.2 需要 17 |
| Android SDK | platform 34 + build-tools 34.0.0 | 通过 `ANDROID_HOME` 指定 |
| Gradle | 由 wrapper 拉取 | 已改用国内镜像，官方源在国内常卡住 |

一条命令：

```bash
py mobile/tools/build_apk.py            # release，签名，可直接安装
py mobile/tools/build_apk.py --debug    # debug 包，排查 WebView 问题用
```

脚本内部依次做：`npm run build` → `npx cap copy android` → `gradlew assembleRelease`
→ 收集到 `mobile/release/` 并写 `apk-info.txt`（版本 / 大小 / sha256 / commit / 时间）。

**这一步不能拆开手敲**：只 `npm run build` 忘了 `cap copy`，APK 里跑的仍是上一版
web 产物，现象是"改了没效果"——与 PC 端"源码改了 exe 没重打包"是同一类坑。

### 只调界面时不必打包

```bash
cd mobile && npm run dev      # 浏览器打开，传感器走 devicemotion（需要 https 或 localhost）
```

真机上要连本机 dev server，用 `npm run dev` 后会打印局域网地址，但 http 局域网地址
拿不到传感器（非安全上下文），只能看静态布局。

## 安装

APK 传到手机后点开安装，允许"安装未知来源应用"。应用 ID `cn.soulplayer.app`，
首次启动是授权说明页。Android 10+ 会在启动后申请一次「身体活动识别」权限
（`ACTIVITY_RECOGNITION`，供硬件步检测作跑步提示的第二确认源）；拒绝不影响使用，
只是少了乘车误报抑制。其余传感器不需要运行时授权。

最低 Android 5.1（minSdk 22），目标 SDK 34。

## 签名

`android/soulplayer-release.jks` + `android/keystore.properties`，**两者都不进版本库**；
签名口令由开发者自行保管，不写入文档、代码或任何提交。
`app/build.gradle` 只在签名时从本地 `keystore.properties` 读取口令字段。

> 这两个文件必须单独备份。换了密钥就无法给已安装的 APK 出升级包——
> 签名不一致时系统会直接拒绝覆盖安装，只能先卸载（本地曲库会一起丢掉）。

## 图标与启动图

`py mobile/tools/make_icons.py`，复用 PC 端 `tools/make_icon.py` 的五根波形柱，
配色换成手机端的暖琥珀（`#c4956a` on `#171513`），一次生成 5 个密度的
launcher / round / adaptive-foreground 与横竖屏启动图。

自适应图标的前景内容收在 52%（安全区是 66%）——放太满会在圆形遮罩下切掉两端柱子。

## 目录

```
mobile/
├── src/
│   ├── core/          判定链（平台无关，可用合成 IMU 单测）
│   ├── platform/      传感器 / 播放器 / 存储 / 通道选择
│   └── ui/            界面与示波器
├── android/           Capacitor 生成的工程（含自定义传感器插件）
├── tests/             vitest 用例 + 合成数据 + 标定探针
├── tools/             build_apk.py / make_icons.py / 截图与探针脚本
└── release/           APK 产物（不进版本库）
```
