cmake_minimum_required(VERSION 3.10)
project(PyChanLib)

# 设置 C 标准
set(CMAKE_C_STANDARD 11)

# 查找 Python 开发包
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# 获取所有的 .c 文件
file(GLOB_RECURSE C_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.c")

# 为每个 C 文件创建一个共享库
foreach(SOURCE ${C_SOURCES})
    # 获取文件名（不含扩展名）
    get_filename_component(TARGET_NAME ${SOURCE} NAME_WE)
    
    # 创建共享库目标
    add_library(${TARGET_NAME} SHARED ${SOURCE})
    
    # 设置输出名称，确保与 Python 模块命名约定一致
    set_target_properties(${TARGET_NAME} PROPERTIES 
        PREFIX ""  # 移除 lib 前缀
        SUFFIX ".so"  # Linux/Unix 使用 .so
    )
    
    if(WIN32)
        set_target_properties(${TARGET_NAME} PROPERTIES SUFFIX ".pyd")  # Windows 使用 .pyd
    endif()
    
    # 添加 Python 头文件路径
    target_include_directories(${TARGET_NAME} PRIVATE 
        ${Python3_INCLUDE_DIRS}
    )
    
    # 链接 Python 库
    target_link_libraries(${TARGET_NAME} PRIVATE 
        ${Python3_LIBRARIES}
    )
endforeach()

# 安装规则
install(TARGETS ${ALL_TARGETS}
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
) 