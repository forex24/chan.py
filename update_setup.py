import os

def find_pyx_files(base_dir):
    pyx_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.pyx'):
                pyx_files.append(os.path.join(root, file))
    return pyx_files

def update_setup_py(pyx_files):
    setup_content = """
from setuptools import setup, find_packages
from Cython.Build import cythonize

setup(
    name="YourProjectName",
    version="0.1",
    packages=find_packages(),
    ext_modules=cythonize([
        {pyx_files}
    ], compiler_directives={{'language_level': "3"}}),
    zip_safe=False,
)
"""
    pyx_files_str = ',\n        '.join(f'"{file}"' for file in pyx_files)
    setup_content = setup_content.format(pyx_files=pyx_files_str)

    with open('setup.py', 'w') as f:
        f.write(setup_content)

if __name__ == "__main__":
    base_dir = '.'  # 当前目录，你可以根据需要修改
    pyx_files = find_pyx_files(base_dir)
    update_setup_py(pyx_files)
    print(f"Found {len(pyx_files)} .pyx files and updated setup.py")