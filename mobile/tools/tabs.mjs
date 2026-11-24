/** 截取三个标签页的内容，检查 sheet 面板渲染。 */
import puppeteer from 'puppeteer-core';
import { existsSync } from 'node:fs';
const edge = ['C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'].find(existsSync);
const browser = await puppeteer.launch({ executablePath: edge, headless: true, args: ['--disable-gpu', '--autoplay-policy=no-user-gesture-required'] });
const page = await browser.newPage();
await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 2 });
await page.goto('http://127.0.0.1:4179/', { waitUntil: 'networkidle0' });
const btn = await page.$('button.accent');
if (btn) await btn.click();
await new Promise((r) => setTimeout(r, 2400));
for (const tab of ['曲库', '设置']) {
  const t = await page.$$eval('.tab', (els, name) => {
    const el = els.find((e) => e.textContent === name);
    if (el) el.click();
    return !!el;
  }, tab);
  if (!t) continue;
  await new Promise((r) => setTimeout(r, 300));
  await page.screenshot({ path: `_shot-tab-${tab}.png` });
  console.log('OK', tab);
}
await browser.close();
