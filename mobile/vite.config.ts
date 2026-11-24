import { defineConfig } from 'vite';

// 手机端构建配置。
// 关键点：
//  1. base 用相对路径，便于部署到任意子路径（含 WorkBuddy 站点托管）。
//  2. server.host 开 0.0.0.0，局域网内手机可直连调试（注意：非 https 下
//     DeviceMotion 在局域网 IP 上不可用，实测请用线上链接或 localhost 端口转发）。
//  3. fs.allow 放开到上一层，让 src 能直接 import ../contracts/*.json。
export default defineConfig({
  base: './',
  server: {
    host: '0.0.0.0',
    port: 5178,
    fs: { allow: ['..'] },
  },
  preview: { host: '0.0.0.0', port: 4178 },
  build: {
    target: 'es2020',
    outDir: 'dist',
    emptyOutDir: true,
    assetsInlineLimit: 8192,
  },
});
