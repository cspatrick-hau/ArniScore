from PyInstaller.utils.hooks import collect_data_files
import os

block_cipher = None

datas = [
    ('models/object_detection.pt', 'models'),
    ('models/convlstm_v3.h5', 'models'),
] + collect_data_files('ultralytics') \
  + collect_data_files('torch') \
  + collect_data_files('cv2') \
  + collect_data_files('numpy') \
  + collect_data_files('fpdf')

a = Analysis(
    ['main.py'],
    pathex=['D:\\Werk\\Arnis Desktop'],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'torch',
        'torchvision',
        'cv2',
        'numpy',
        'fpdf',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
        'ultralytics',
        'tensorflow',
        'keras',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='ArnisStrikeDetector',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    icon='ArniScore_Logo.ico'
)
