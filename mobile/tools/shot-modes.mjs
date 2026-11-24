/**
 * 模式切换的视觉验证：日常手势页 → 跑步手势页 → 跑步设置页。
 *
 * 为什么必须截图而不是只看 DOM：#409 的那些警示条、位置档按钮、芯片配色
 * 只有在真实渲染里才能看出是否被截断或对比度不足。
 */
import puppeteer from 'puppeteer-core';
import { existsSync } from 'node:fs';

const edge = ['C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'].find(existsSync);
const base = process.argv[3] ?? 'http://127.0.0.1:4181/';
const prefix = process.argv[2] ?? '_mode';

const browser = await puppeteer.launch({
  executablePath: edge,
  headless: true,
  args: ['--disable-gpu', '--autoplay-policy=no-user-gesture-required'],
});
const page = await browser.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
page.on('console', (m) => {
  if (m.type() === 'error') errors.push('console: ' + m.text());
});
await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 2 });
await page.goto(base, { waitUntil: 'networkidle0' });

const btn = await page.$('button.accent');
if (btn) await btn.click();
await new Promise((r) => setTimeout(r, 2600));

await page.screenshot({ path: `${prefix}-1-still.png` });
console.log('still gestures ok');

const chipText = await page.$eval('.mode-chip', (el) => el.textContent);
await page.click('.mode-chip');
await new Promise((r) => setTimeout(r, 400));
await page.screenshot({ path: `${prefix}-2-run.png` });
const runChip = await page.$eval('.mode-chip', (el) => el.textContent);
const runRows = await page.$$eval('.sheet .row .name', (els) => els.map((e) => e.textContent));
const banner = await page.$eval('.sheet .banner', (el) => el.textContent).catch(() => null);
const segments = await page.$$eval('.seg button', (els) => els.map((e) => e.textContent));
console.log('chip:', JSON.stringify(chipText), '->', JSON.stringify(runChip));
console.log('run rows:', JSON.stringify(runRows));
console.log('segments:', JSON.stringify(segments));
console.log('banner:', banner);

const settingsTab = await page.$$eval('.tab', (els) => {
  const el = els.find((e) => e.textContent === '设置');
  if (el) el.click();
  return !!el;
});
if (settingsTab) {
  await new Promise((r) => setTimeout(r, 250));
  await page.screenshot({ path: `${prefix}-3-run-settings.png` });
  const disabled = await page.$$eval('.btn', (els) =>
    els.map((e) => ({ text: e.textContent, disabled: e.disabled })),
  );
  console.log('settings buttons:', JSON.stringify(disabled));
}

await page.click('.mode-chip');
await new Promise((r) => setTimeout(r, 350));
const backChip = await page.$eval('.mode-chip', (el) => el.textContent);
const backRows = await page.$$eval('.sheet .row .name', (els) => els.map((e) => e.textContent));
console.log('back chip:', JSON.stringify(backChip), 'rows:', JSON.stringify(backRows));
console.log('errors:', errors.length ? JSON.stringify(errors) : 'none');

await browser.close();
