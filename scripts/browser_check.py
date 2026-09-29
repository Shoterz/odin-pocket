"""Behavioral browser checks and real screenshots for the local workbench."""
import argparse
from pathlib import Path
from playwright.sync_api import sync_playwright

p=argparse.ArgumentParser()
p.add_argument('--url',default='http://127.0.0.1:8766')
p.add_argument('--screenshots',default='submission/screenshots')
p.add_argument('--require-model',action='store_true')
p.add_argument('--chromium',help='Optional Chromium executable; otherwise uses Playwright installation')
a=p.parse_args()
out=Path(a.screenshots);out.mkdir(parents=True,exist_ok=True)
with sync_playwright() as pw:
    browser=pw.chromium.launch(headless=True,**({'executable_path':a.chromium} if a.chromium else {}))
    page=browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1)
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto(a.url)
    page.wait_for_function("() => document.getElementById('connection').textContent !== 'Connecting…'")
    assert page.title().startswith('ODIN Pocket')
    if a.require_model:
        assert 'Checkpoint loaded' in page.locator('#connection').inner_text()
        page.locator('#length').focus()
        page.keyboard.press('Home')
        page.keyboard.press('ArrowRight')
        page.locator('#generate').click()
        page.wait_for_function("() => !document.getElementById('generate').disabled",timeout=180000)
        assert 'tokens/s' in page.locator('#generation-stats').inner_text()
        assert page.locator('#tokens button').count()>0
        page.locator('#tokens button').first.focus()
        page.keyboard.press('Enter')
        assert 'probability' in page.locator('#token-caption').inner_text()
    page.screenshot(path=str(out/'01-workbench.png'),full_page=True)
    page.locator('[data-view=compare]').click()
    if a.require_model:
        page.locator('#compare').click()
        page.wait_for_function("() => !document.getElementById('compare').disabled",timeout=180000)
        assert page.locator('.candidate-result').count()==2
    page.screenshot(path=str(out/'02-comparison.png'),full_page=True)
    page.locator('[data-view=evidence]').click()
    page.locator('#benchmarks tr').first.wait_for()
    assert page.locator('#benchmarks tr').count()==5
    page.screenshot(path=str(out/'03-evidence.png'),full_page=True)
    # Regression: empty evidence removes an old rendered curve.
    page.evaluate("chart([{tokens:0,dev_loss:8},{tokens:100,dev_loss:4}]);chart([])")
    assert page.locator('#loss-chart svg').count()==0
    assert 'No matching' in page.locator('#loss-chart').inner_text()
    if a.require_model:
        page.locator('#reload').click()
        page.wait_for_function("() => !document.getElementById('reload').disabled",timeout=180000)
        assert page.locator('.candidate-result').count()==0
        assert page.locator('#tokens button').count()==0
    page.set_viewport_size({'width':390,'height':844})
    page.locator('[data-view=write]').click()
    assert page.locator('#reload').is_visible()
    assert not page.evaluate('document.documentElement.scrollWidth>window.innerWidth')
    page.screenshot(path=str(out/'04-mobile.png'),full_page=True)
    assert not errors,errors
    browser.close()
print('Browser checks passed. Screenshots:',out)
