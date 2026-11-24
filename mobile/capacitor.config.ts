import type { CapacitorConfig } from '@capacitor/cli';

/**
 * Capacitor 配置。
 *
 * 关键点是 androidScheme='https'：WebView 里页面地址是 https://localhost，
 * 因此 window.isSecureContext 为 true，web 通道的传感器/加密存储等能力判定
 * 与线上 PWA 保持一致，不需要为 APK 单独写一套分支。
 */
const config: CapacitorConfig = {
  appId: 'cn.soulplayer.app',
  appName: 'SoulPlayer',
  webDir: 'dist',
  android: {
    // 允许 WebView 里的 Audio 元素继续播放（切到后台/锁屏时不强杀）
    allowMixedContent: false,
  },
  server: {
    androidScheme: 'https',
  },
};

export default config;
