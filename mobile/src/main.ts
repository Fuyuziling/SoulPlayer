import { App } from './ui/app';

/**
 * 入口。只做一件事：把 App 挂到 #app 上。
 * 真正的装配（引擎 / 传感器 / 播放器）都在 App 内部完成，
 * 这样以后要做"遥控 PC"之类的第二种运行模式时，只需要换一个 App 实现。
 */
const root = document.getElementById('app');
if (root) {
  const app = new App(root);
  void app.boot();
} else {
  document.body.textContent = '页面结构缺失：#app 未找到';
}
