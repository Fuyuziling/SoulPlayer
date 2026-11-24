/**
 * 视觉验证脚本：用本机 Edge 截图，供开发时自查 UI。
 * 用法：node tools/shot.mjs [输出前缀] [页面URL]
 * 之所以不用 msedge --headless --screenshot：那个模式在页面 load 事件就拍照，
 * 而本应用的启动门是 IndexedDB 读取之后才渲染的，每次都拍到黑屏。
 */
import puppeteer from 'puppeteer-core';
import { existsSync } from 'node:fs';

const EDGE_PATHS = [
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  'C:/Program Files/Microsoft/Edge/Application/msedge.exe',
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
];

const edge = EDGE_PATHS.find((p) => existsSync(p));
if (!edge) {
  console.error('找不到 Edge/Chrome');
  process.exit(1);
}

const prefix = process.argv[2] ?? 'shot';
const url = process.argv[3] ?? 'http://127.0.0.1:4179/';

const browser = await puppeteer.launch({
  executablePath: edge,
  headless: true,
  args: ['--disable-gpu', '--no-first-run'],
});

try {
  const page = await browser.newPage();
  await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 2 });
  page.on('pageerror', (e) => console.error('[pageerror]', e.message));
  page.on('console', (m) => {
    if (m.type() === 'error') console.error('[console]', m.text());
  });

  await page.goto(url, { waitUntil: 'networkidle0', timeout: 20000 });
  await new Promise((r) => setTimeout(r, 900));
  await page.screenshot({ path: `${prefix}-1-gate.png` });
  console.log(`OK ${prefix}-1-gate.png`);

  // 点击"启用传感器并开始"：无传感器环境下 MotionSource 会因非 https/不支持而失败，
  // 这里只验证 UI 能不能走到主界面或正确显示错误。
  const clicked = await page
    .evaluate(() => {
      const btn = [...document.querySelectorAll('button')].find((b) => b.textContent?.includes('启用传感器'));
      if (btn) {
        btn.click();
        return true;
      }
      return false;
    })
    .catch(() => false);

  if (clicked) {
    await new Promise((r) => setTimeout(r, 2600));
    await page.screenshot({ path: `${prefix}-2-main.png` });
    console.log(`OK ${prefix}-2-main.png`);
  }
} finally {
  await browser.close();
}
