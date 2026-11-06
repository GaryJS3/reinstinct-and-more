const { chromium } = require('playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');
(async () => {
  const browser = await chromium.launch({channel:'msedge', headless:true});
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', error => {errors.push(error.message);console.error(error.message)});
    const html = fs.readFileSync(process.argv[2] || 'target/dashboard-review.html', 'utf8');
    const gpu = pci => ({pci_address:pci, vendor:'AMD', name:'Test GPU', controls:{helper_installed:true,power_limit_writable:true},power:{minimum_watts:100,maximum_watts:300,limit_watts:200}});
    let statusFail=false, runsFail=false;
    const config={persisted:true,path:'/test/server.json',saved:{max_seq:4096,gpu_max_temp_c:85},effective:{max_seq:4096,gpu_max_temp_c:85}};
    await page.route('http://dashboard.test/**', async route => {
      const path=new URL(route.request().url()).pathname;
      if(path==='/')return route.fulfill({contentType:'text/html',body:html});
      if((path==='/api/status'&&statusFail)||(path==='/api/runs'&&runsFail)||path==='/api/runs/1')return route.fulfill({status:503,json:{error:{message:'test failure'}}});
      const payloads={
        '/api/status':{status:'ready',service:{version:'test'},model:{id:'Test'},metrics:{},worker:{},gpu:{},uptime_seconds:1},
        '/api/gpus':{gpus:{a:gpu('0000:01:00.0'),b:gpu('0000:02:00.0')}},
        '/api/config':config,
        '/api/runs':{runs:[{id:1,state:'complete',stats:{}}]},
        '/api/logs':{lines:[]},
        '/api/models':{count:1,models:[{id:'Test',path:'/model.gguf',size_gib:1}]}
      };
      return route.fulfill({json:payloads[path]||{}});
    });
    await page.goto('http://dashboard.test/');
    await page.waitForFunction(()=>document.getElementById('phase').textContent==='ready'&&document.querySelector('[data-config="max_seq"]'));
    assert.equal(await page.locator('style').count(),3);
    assert.equal(await page.locator('.gpu-meters').evaluate(el=>getComputedStyle(el).display),'grid');
    await page.locator('#amdPanel summary').click();
    await page.selectOption('#amdPci','0000:02:00.0');
    await page.evaluate(()=>refreshAmd());
    assert.equal(await page.inputValue('#amdPci'),'0000:02:00.0');
    await page.waitForFunction(()=>!document.getElementById('amdWatts').disabled);
    await page.locator('#amdWatts').fill('225');
    await page.locator('#amdPower').focus();
    await page.waitForTimeout(2200);
    assert.equal(await page.inputValue('#amdWatts'),'225');
    await page.locator('#configPanel summary').click();
    await page.locator('[data-config="max_seq"]').fill('8192');
    await page.evaluate(()=>loadConfig());
    assert.equal(await page.inputValue('[data-config="max_seq"]'),'8192');
    await page.locator('#thermalPanel summary').click();
    await page.locator('[data-thermal="gpu_max_temp_c"]').fill('90');
    await page.click('#configSave');
    await page.waitForFunction(()=>document.getElementById('configSave').textContent==='Save');
    assert.equal(await page.inputValue('[data-thermal="gpu_max_temp_c"]'),'90');
    await page.locator('#modelsPanel summary').click();
    await page.click('#scanModels');
    await page.evaluate(()=>refresh());
    assert.equal(await page.locator('#modelList button').isDisabled(),true);
    await page.locator('#runsPanel summary').click();
    await page.waitForSelector('tr[data-id="1"]');
    await page.locator('tr[data-id="1"]').focus();
    await page.keyboard.press('Enter');
    await page.waitForFunction(()=>document.getElementById('detail').open);
    assert.equal(await page.locator('#detailTitle').textContent(),'Run unavailable');
    await page.keyboard.press('Escape');
    runsFail=true;
    await page.evaluate(()=>refresh());
    assert.equal(await page.locator('#phase').textContent(),'ready');
    statusFail=true;
    await page.evaluate(()=>refresh());
    assert.match(await page.locator('#phase').textContent(),/stale/);
    await page.setViewportSize({width:390,height:844});
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
    await page.screenshot({path:'target/dashboard-mobile.png',fullPage:true});
    assert.deepEqual(errors,[]);
    console.log('Passed: stylesheet parsing, GPU selection and draft persistence, independent settings drafts, loaded-model state, keyboard detail/error handling, partial failures, stale status, mobile overflow, browser errors.');
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1});

