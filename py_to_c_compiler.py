import os
import sys
from Cython.Build import cythonize
from setuptools import setup
from distutils.core import Extension
import shutil

def compile_py_to_c(py_file_path):
    """将单个 Python 文件编译成 C 文件"""
    try:
        # 使用 cythonize 编译文件
        extension = Extension(
            name=os.path.splitext(os.path.basename(py_file_path))[0],
            sources=[py_file_path]
        )
        setup(
            ext_modules=cythonize(
                extension,
                compiler_directives={'language_level': "3"}
            ),
            script_args=['build_ext', '--inplace']
        )
        print(f"成功编译: {py_file_path}")
        return True
    except Exception as e:
        print(f"编译失败 {py_file_path}: {str(e)}")
        return False

def process_directory(directory):
    """递归处理目录中的所有 .py 文件"""
    compiled_files = []
    failed_files = []
    
    # 定义要处理的文件白名单
    main_dir_whitelist = {'Chan.py', 'ChanConfig.py'}
    # 定义要忽略的目录
    ignore_dirs = {'Debug', 'DataProcess'}
    
    for root, dirs, files in os.walk(directory):
        # 从遍历列表中移除要忽略的目录
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        for file in files:
            if not file.endswith('.py'):
                continue
                
            py_file_path = os.path.join(root, file)
            
            # 跳过当前脚本文件
            if os.path.samefile(py_file_path, __file__):
                continue
            
            # 检查文件是否在主目录中
            is_in_main_dir = os.path.dirname(py_file_path) == directory
            
            # 如果在主目录中，只处理白名单文件
            if is_in_main_dir and file not in main_dir_whitelist:
                continue
                
            print(f"正在处理: {py_file_path}")
            success = compile_py_to_c(py_file_path)
            
            if success:
                compiled_files.append(py_file_path)
            else:
                failed_files.append(py_file_path)
    
    return compiled_files, failed_files

def cleanup_build_files():
    """清理编译过程中生成的临时文件"""
    # 清理 build 目录
    if os.path.exists('build'):
        shutil.rmtree('build')
    
    # 清理其他临时文件
    for file in os.listdir('.'):
        if file.endswith(('.c', '.so', '.pyd')):
            os.remove(file)

def main():
    # 确保已安装 Cython
    try:
        import Cython
    except ImportError:
        print("请先安装 Cython: pip install cython")
        sys.exit(1)

    # 获取当前目录
    current_dir = os.getcwd()
    print(f"开始处理目录: {current_dir}")
    
    # 编译文件
    compiled_files, failed_files = process_directory(current_dir)
    
    # 打印结果
    print("\n编译完成!")
    print(f"成功编译的文件数: {len(compiled_files)}")
    print(f"失败的文件数: {len(failed_files)}")
    
    if failed_files:
        print("\n失败的文件:")
        for file in failed_files:
            print(f"- {file}")
    
    # 清理临时文件
    cleanup_build_files()

if __name__ == "__main__":
    main() 