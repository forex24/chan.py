import os
import shutil
import subprocess

def main():
    # 0. 卸载已存在的 pychan 包
    try:
        subprocess.run(['pip', 'uninstall', 'pychan', '-y'], check=True)
        print("已卸载现有的pychan包")
    except subprocess.CalledProcessError as e:
        print(f"卸载pychan包失败或包不存在: {e}")
        # 这里我们继续执行，因为包可能本来就不存在

    # 1. 删除build目录及其子目录（如果存在）
    if os.path.exists('build'):
        shutil.rmtree('build')
        print("已删除build目录")

    # 2. 执行merak cythonize命令
    try:
        subprocess.run(['merak', 'cythonize', 'pychan', 'build'], check=True)
        print("merak cythonize 命令执行成功")
    except subprocess.CalledProcessError as e:
        print(f"执行merak cythonize失败: {e}")
        return

    # 3. 进入build目录
    os.chdir('build')
    print("已进入build目录")

    # 4. 创建setup.py文件
    setup_content = '''import setuptools

setuptools.setup(
    name="pychan",
    version="0.1.0",
    packages=["pychan"],
    include_package_data=True,
    package_data={"pychan": ["*"]},
    zip_safe=False
)
'''
    
    with open('setup.py', 'w') as f:
        f.write(setup_content)
    print("已创建setup.py文件")

    # 5. 执行 python setup.py bdist_wheel
    try:
        subprocess.run(['python', 'setup.py', 'bdist_wheel'], check=True)
        print("wheel包构建成功")
    except subprocess.CalledProcessError as e:
        print(f"构建wheel包失败: {e}")
        return

    # 6. 进入dist目录并安装wheel包
    os.chdir('dist')
    try:
        subprocess.run(['pip', 'install', 'pychan-0.1.0-py3-none-any.whl'], check=True)
        print("wheel包安装成功")
    except subprocess.CalledProcessError as e:
        print(f"安装wheel包失败: {e}")
        return

if __name__ == "__main__":
    main()
