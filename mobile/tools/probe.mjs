import puppeteer from 'puppeteer-core';
import { existsSync } from 'node:fs';
const edge = ['C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'].find(existsSync);
const browser = await puppeteer.launch({ executablePath: edge, headless: true, args: ['--disable-gpu'] });
const page = await browser.newPage();
page.on('pageerror', (e) => console.error('[pageerror]', e.message));
page.on('console', (m) => console.log('[console]', m.type(), m.text()));
await page.setViewport({ width: 390, height: 844 });
await page.goto('http://127.0.0.1:4179/', { waitUntil: 'networkidle0' });
await page.evaluate(() => {
  window.addEventListener('error', (e) => console.log('[winerror]', e.message));
});
const btn = await page.$('button.accent');
if (btn) await btn.click();  // 真实点击，带 user activation
for (let i = 0; i < 5; i++) {
  await new Promise((r) => setTimeout(r, 1000));
  const state = await page.evaluate(() => ({
    gate: !!document.querySelector('.gate'),
    main: !!document.querySelector('.topbar'),
    hint: document.querySelector('.gate .hint')?.textContent?.slice(0, 40) ?? null,
    dot: !!document.querySelector('.dot'),
  }));
  console.log(i, JSON.stringify(state));
  if (state.main) break;
}
await browser.close();
