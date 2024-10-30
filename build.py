import os
import subprocess
import platform
import shutil

def build_project():
    # 创建构建目录
    if not os.path.exists('build'):
        os.makedirs('build')
    
    # 进入构建目录
    os.chdir('build')
    
    # 配置 CMake
    cmake_configure = subprocess.run(['cmake', '..'], check=True)
    if cmake_configure.returncode != 0:
        print("CMake 配置失败")
        return False
    
    # 构建项目
    cmake_build = subprocess.run(['cmake', '--build', '.'], check=True)
    if cmake_build.returncode != 0:
        print("构建失败")
        return False
    
    print("构建成功!")
    return True

def copy_shared_libs():
    """复制共享库到项目根目录"""
    extension = '.pyd' if platform.system() == 'Windows' else '.so'
    build_dir = 'build'
    
    # 遍历构建目录查找共享库
    for root, _, files in os.walk(build_dir):
        for file in files:
            if file.endswith(extension):
                src_path = os.path.join(root, file)
                dst_path = os.path.join('..', file)
                try:
                    shutil.copy2(src_path, dst_path)
                    print(f"已复制: {file}")
                except Exception as e:
                    print(f"复制失败 {file}: {str(e)}")

if __name__ == "__main__":
    if build_project():
        copy_shared_libs() 