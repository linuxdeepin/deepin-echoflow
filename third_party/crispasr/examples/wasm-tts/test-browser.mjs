// Headless-Chrome harness: starts the COOP/COEP server, loads the demo in real Chromium (with
// SharedArrayBuffer via cross-origin isolation), clicks Speak, and reports whether synthesis
// completes — so the browser path can be debugged without a human. Run: node test-browser.mjs
import { spawn } from 'node:child_process';
import puppeteer from 'puppeteer-core';

const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const TIMEOUT_MS = Number(process.env.TIMEOUT_MS || 120000);

const srv = spawn('node', ['server.mjs'], { cwd: import.meta.dirname, stdio: 'ignore' });
const sleep = ms => new Promise(r => setTimeout(r, ms));
await sleep(700);

const browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: 'new',
    args: ['--no-sandbox', '--disable-gpu', '--enable-features=SharedArrayBuffer',
        '--autoplay-policy=no-user-gesture-required']
});
try {
    const page = await browser.newPage();
    page.on('console', m => console.log('[page]', m.text()));
    page.on('pageerror', e => console.log('[pageerror]', e.message));
    page.on('workererror', e => console.log('[workererror]', e.message));

    await page.goto('http://localhost:8791/', { waitUntil: 'load', timeout: 30000 });
    console.log('crossOriginIsolated:', await page.evaluate(() => self.crossOriginIsolated));

    await page.click('#speak');
    const t0 = Date.now();
    let result = 'TIMEOUT (still hung at last log line)';
    while (Date.now() - t0 < TIMEOUT_MS) {
        const txt = await page.$eval('#log', el => el.textContent);
        if (/✓ \d+ samples/.test(txt)) { result = 'SUCCESS ' + (/✓ [^\n]*/.exec(txt) || [''])[0]; break; }
        if (/✗/.test(txt)) { result = 'ERROR: ' + txt.split('\n').filter(l => l.includes('✗')).join(' | '); break; }
        await sleep(1000);
    }
    console.log('\n=== RESULT after ' + ((Date.now() - t0) / 1000).toFixed(0) + 's:', result);
    console.log('=== FINAL PAGE LOG ===\n' + await page.$eval('#log', el => el.textContent));
} finally {
    await browser.close();
    srv.kill();
}
process.exit(0);
