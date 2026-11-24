package cn.soulplayer.app;

import android.os.Bundle;

import com.getcapacitor.BridgeActivity;

public class MainActivity extends BridgeActivity {

    /**
     * 自定义插件必须在 super.onCreate() 之前注册：
     * BridgeActivity.onCreate 会构建 Bridge 并初始化插件表，
     * 之后注册的插件在 WebView 侧找不到（表现为调用报 "not implemented"，
     * 而不是报缺少插件，很容易误判成 JS 写错）。
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        registerPlugin(SoulSensorsPlugin.class);
        super.onCreate(savedInstanceState);
    }
}
