# -*- mode: python ; coding: utf-8 -*-
import os
import sys

# Get the fast_ssim package location
fast_ssim_path = os.path.join(sys.prefix, 'lib', 'site-packages', 'fast_ssim')
ssim_dll_path = os.path.join(fast_ssim_path, 'resources', 'ssim.dll')

# Get the wordninja_enhanced package location
wordninja_path = os.path.join(sys.prefix, 'lib', 'site-packages', 'wordninja_enhanced')
wordninja_resources = os.path.join(wordninja_path, 'resources')

a = Analysis(
    ['videocr_glm_cli.py'],
    pathex=[],
    binaries=[
        (ssim_dll_path, 'fast_ssim/resources'),
    ],
    datas=[
        (wordninja_resources, 'wordninja_enhanced/resources'),
    ],
    hiddenimports=['fast_ssim._core'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='videocr_glm_cli',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
