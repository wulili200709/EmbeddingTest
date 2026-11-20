// Real-browser smoke checks for the local four-point review tool.
// Usage: node qa_lab_line_browser.cjs <playwright package path> <report folder>
const { chromium } = require(process.argv[2]);
const fs = require('fs');
const path = require('path');
const {pathToFileURL} = require('url');
const assert = require('assert/strict');
(async()=>{
  const folder=path.resolve(process.argv[3]);
  const qa=path.join(folder,'qa');fs.mkdirSync(qa,{recursive:true});
  const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe'});
  try{
    const page=await browser.newPage({viewport:{width:1440,height:1100},deviceScaleFactor:1});
    const errors=[];page.on('pageerror',error=>errors.push(error.message));
    await page.addInitScript(()=>{window.__qaImages=[];const NativeImage=window.Image;window.Image=function(...args){const image=new NativeImage(...args);window.__qaImages.push(image);return image;};});
    await page.goto(pathToFileURL(path.join(folder,'four_point_review.html')).href);
    await page.waitForFunction(()=>document.querySelector('#resultsBody').children.length===84);
    await page.waitForFunction(()=>window.__qaImages?.length&&window.__qaImages.every(image=>image.complete&&image.naturalWidth>0));
    await page.waitForFunction(()=>!document.querySelector('#statusMessage').textContent.includes('无法读取'));
    const sourceData=await page.locator('#labLineData').textContent().then(JSON.parse);
    const bodyMode=sourceData.meta?.method==='body_outline';
    if(bodyMode){
      assert.equal(await page.locator('#showLegacy').isChecked(),false);
      assert.equal(await page.locator('#showPrevious').isChecked(),true);
      assert.equal(await page.locator('#showPrevious').isVisible(),true);
      assert.match(await page.locator('#bodyMethodNote').innerText(),/左右稳定段定线/);
      assert.match(await page.locator('#manualInstruction').innerText(),/无需人工四点/);
      assert.match(await page.locator('#progressTitle').innerText(),/可选/);
      const validInitial=sourceData.images.flatMap((im,i)=>im.points.map(p=>({i,p}))).find(({p})=>p.auto.status!=='failed'&&Number.isFinite(p.auto.distance_px)&&p.previous_auto?.upper?.segment&&p.previous_auto?.lower?.segment&&['upper','lower'].some(side=>JSON.stringify(p.auto[side].segment)!==JSON.stringify(p.previous_auto[side].segment)));
      assert.ok(validInitial,'need a changed measured candidate so both overlay colors are visible');
      await page.locator('#imageSelect').selectOption(String(validInitial.i));
      await page.getByRole('button',{name:validInitial.p.name,exact:true}).click();
      assert.notEqual(await page.locator('#autoPx').innerText(),'— px');
      const canvasWithPrevious=await page.locator('#focusCanvas').screenshot();
      await page.locator('#showPrevious').uncheck();
      const canvasWithoutPrevious=await page.locator('#focusCanvas').screenshot();
      assert.ok(!canvasWithPrevious.equals(canvasWithoutPrevious),'previous overlay must change the canvas');
      await page.locator('#showPrevious').check();
      const {PNG}=require(path.join(path.dirname(process.argv[2]),'pngjs'));
      const canvasPixels=PNG.sync.read(await page.locator('#focusCanvas').screenshot()).data;
      const canvasColors={orange:0,cyan:0};for(let i=0;i<canvasPixels.length;i+=4){if(canvasPixels[i]===255&&canvasPixels[i+1]===155&&canvasPixels[i+2]===69)canvasColors.orange++;if(canvasPixels[i]===0&&canvasPixels[i+1]===215&&canvasPixels[i+2]===209)canvasColors.cyan++;}
      assert.ok(canvasColors.orange>5&&canvasColors.cyan>5,JSON.stringify(canvasColors));
    }
    assert.equal(await page.locator('#reviewCount').innerText(),'0 / 84');
    assert.equal(await page.locator('#manualPx').innerText(),'— px');
    assert.deepEqual(await page.locator('.point-button').allTextContents(),['L7','L6','L5','L4','L3','L2','L1','U7','U6','U5','U4','U3','U2','U1']);
    assert.deepEqual(await page.locator('#resultsBody tr').first().locator('td').allTextContents().then(a=>a.slice(0,2)),['1.png','U1']);
    for(const [name,row,col] of [['L7','top',1],['L1','top',7],['U7','bottom',1],['U1','bottom',7]]){
      await page.getByRole('button',{name,exact:true}).click();
      const selected=await page.evaluate(()=>window.LabLineCompare.getCurrent().p);
      assert.deepEqual([selected.name,selected.physical_row,selected.physical_col],[name,row,col]);
    }
    const failure=sourceData.images.flatMap((im,i)=>im.points.map(p=>({im,i,p}))).find(e=>e.p.auto?.status==='failed'&&e.p.auto.upper?.segment&&e.p.auto.lower?.segment);
    if(failure){
      await page.locator('#imageSelect').selectOption(String(failure.i));
      await page.getByRole('button',{name:failure.p.name,exact:true}).click();
      assert.equal(await page.locator('#autoPx').innerText(),'— px');
      assert.match(await page.locator('#autoMm').innerText(),/无有效自动尺寸/);
      const failedRecord=await page.evaluate(({image,point})=>window.LabLineCompare.exportObject().records.find(r=>r.image===image&&r.point===point),{image:failure.im.name,point:failure.p.name});
      assert.equal(failedRecord.auto_distance_px,null);
      assert.equal(failedRecord.auto_has_valid_distance,false);
      assert.equal(failedRecord.manual_distance_px,null);
      assert.match(await page.locator('#autoStatus').innerText(),/红色虚线.*不代表正确边界/);
      const {PNG:FailurePNG}=require(path.join(path.dirname(process.argv[2]),'pngjs'));
      const failedPixels=FailurePNG.sync.read(await page.locator('#focusCanvas').screenshot()).data;
      let redDiagnosticPixels=0;for(let i=0;i<failedPixels.length;i+=4){if(failedPixels[i]===245&&failedPixels[i+1]===70&&failedPixels[i+2]===70)redDiagnosticPixels++;}
      assert.ok(redDiagnosticPixels>5,'failed fit must render red diagnostic dashes');
      await page.locator('#focusCanvas').screenshot({path:path.join(qa,'failed_fit_red_diagnostic.png')});
      if(await page.locator('#seedAuto').isEnabled()){
        await page.locator('#seedAuto').click();
        assert.match(await page.locator('#manualSource').innerText(),/原自动结果无有效尺寸/);
        assert.equal(await page.locator('#reviewCount').innerText(),'0 / 84');
        assert.equal(await page.locator('#autoPx').innerText(),'— px');
        await page.locator('#resetPoint').click();
      }
    }
    await page.locator('#imageSelect').selectOption('5');
    await page.locator('.point-button').filter({hasText:/^U1$/}).click();
    assert.equal(await page.locator('#selectionTitle').innerText(),'6.png / U1');
    await page.locator('#seedAuto').click();
    assert.equal(await page.locator('#reviewCount').innerText(),'0 / 84');
    assert.equal(await page.locator('#draftCount').innerText(),'1');
    await page.locator('#confirmManual').click();
    assert.equal(await page.locator('#reviewCount').innerText(),'1 / 84');
    const downloadPromise=page.waitForEvent('download');
    await page.locator('#exportJson').click();
    const download=await downloadPromise;
    const exportFile=path.join(qa,'synthetic_ui_review_export.json');await download.saveAs(exportFile);
    const exported=JSON.parse(fs.readFileSync(exportFile,'utf8'));
    assert.ok(JSON.stringify(exported).includes('6.png'));
    // Reload must preserve the explicitly confirmed record.
    await page.reload();
    await page.waitForFunction(()=>document.querySelector('#reviewCount').textContent==='1 / 84');
    await page.locator('#imageSelect').selectOption('5');
    await page.locator('.point-button').filter({hasText:/^U1$/}).click();
    await page.locator('#resetPoint').click();
    assert.equal(await page.locator('#reviewCount').innerText(),'0 / 84');
    assert.equal(await page.locator('#manualPx').innerText(),'— px');
    await page.locator('#imageSelect').selectOption('0');
    await page.getByRole('button',{name:'L7',exact:true}).click();
    await page.locator('#showCandidates').check();
    await page.locator('details').evaluate(el=>el.open=true);
    await page.waitForFunction(()=>window.__qaImages?.length&&window.__qaImages.every(image=>image.complete&&image.naturalWidth>0));
    if(bodyMode){
      assert.match(await page.locator('#candidateLegend').innerText(),/左右稳定段定线实测(?:内)?点/);
      assert.match(await page.locator('#qualityInfo').innerText(),/左右定线段RMSE/);
      assert.match(await page.locator('#qualityInfo').innerText(),/全轮廓内点RMSE/);
      assert.match(await page.locator('#qualityInfo').innerText(),/全部有效实测点RMSE/);
    }
    await page.screenshot({path:path.join(qa,'four_point_review_desktop.png'),fullPage:true});
    await page.setViewportSize({width:390,height:844});
    await page.screenshot({path:path.join(qa,'four_point_review_mobile.png'),fullPage:true});
    const bodySize=await page.evaluate(()=>({viewport:innerWidth,body:document.body.scrollWidth}));
    assert.ok(bodySize.body<=bodySize.viewport+1,JSON.stringify(bodySize));
    assert.deepEqual(errors,[]);
    const result={page_errors:errors,points:84,manual_initially_empty:true,mapping_checked:true,
      auto_seed_requires_confirmation:true,persistence_checked:true,reset_checked:true,
      export_checked:true,mobile_no_horizontal_body_overflow:true,
      method:sourceData.meta?.method||'line_comparison',body_defaults_checked:bodyMode,
      previous_overlay_and_current_cyan_checked:bodyMode,automatic_without_manual_checked:bodyMode,
      failed_automatic_null_checked:!!failure,failed_fit_red_diagnostic_checked:!!failure,
      note:'Export is a synthetic browser test of an auto-seeded confirmation, not a real human measurement.'};
    fs.writeFileSync(path.join(qa,'browser_qa.json'),JSON.stringify(result,null,2));
    console.log(JSON.stringify(result));
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1});
