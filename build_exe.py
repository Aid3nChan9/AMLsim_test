"""
PyInstaller 打包脚本
使用方法: python build_exe.py
"""

import os
import sys
import subprocess

def main():
    app_dir = os.path.dirname(os.path.abspath(__file__))
    app_script = os.path.join(app_dir, 'aml_app.py')
    output_dir = os.path.join(app_dir, 'dist')

    print("=" * 60)
    print("  AML 反洗钱检测系统 - PyInstaller 打包")
    print("=" * 60)
    print(f"应用脚本: {app_script}")
    print(f"输出目录: {output_dir}")
    print()

    # 检查 PyInstaller
    try:
        import PyInstaller
        print(f"PyInstaller 版本: {PyInstaller.__version__}")
    except ImportError:
        print("正在安装 PyInstaller...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyinstaller'])

    icon_path = os.path.join(app_dir, 'aml_icon.ico')
    icon_arg = []
    if os.path.exists(icon_path):
        icon_arg = ['--icon', icon_path]
        print(f"使用图标: {icon_path}")
    else:
        print("未找到 aml_icon.ico，使用默认图标")

    # 构建 PyInstaller 命令
    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--name', 'AML_Detector',
        '--onefile',           # 打包为单文件
        '--windowed',          # 不显示控制台窗口（GUI模式）
        '--clean',             # 清理缓存
        '--noconfirm',         # 覆盖输出目录
        '--add-data', f'{app_script};.',  # 包含源文件
    ]

    # 隐藏导入（PyInstaller 可能漏掉的关键依赖）
    hidden_imports = [
        'pandas', 'numpy', 'networkx', 'sklearn', 'sklearn.ensemble',
        'sklearn.ensemble._forest', 'sklearn.tree', 'sklearn.utils',
        'sklearn.metrics', 'sklearn.metrics.cluster',
        'imblearn', 'imblearn.combine', 'imblearn.over_sampling',
        'matplotlib', 'matplotlib.backends.backend_agg',
        'seaborn', 'shap', 'PIL', 'PIL.Image', 'PIL.ImageTk',
        'xgboost', 'joblib', 'scipy', 'threading',
    ]
    for mod in hidden_imports:
        cmd.extend(['--hidden-import', mod])

    # 排除不需要的模块减小体积
    excludes = [
        'tkinter.test', 'unittest', 'test', 'pydoc',
        'matplotlib.tests', 'pandas.tests', 'numpy.tests',
    ]
    for mod in excludes:
        cmd.extend(['--exclude-module', mod])

    cmd.append(app_script)

    print("\n执行命令:")
    print(' '.join(cmd))
    print()
    print("打包中，请耐心等待（可能需要 5-10 分钟）...")
    print()

    subprocess.check_call(cmd, cwd=app_dir)

    exe_path = os.path.join(output_dir, 'AML_Detector.exe')
    if os.path.exists(exe_path):
        size_mb = os.path.getsize(exe_path) / (1024 * 1024)
        print(f"\n{'=' * 60}")
        print(f"  打包成功! ✓")
        print(f"  输出文件: {exe_path}")
        print(f"  文件大小: {size_mb:.1f} MB")
        print(f"{'=' * 60}")
    else:
        print("\n打包失败，请检查错误信息")

if __name__ == '__main__':
    main()
