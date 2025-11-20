/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#define private public
#include "task.h"
#include "task_executor.h"

using namespace testing;
using namespace Ascc;

namespace  {
std::stringstream buffer;
std::streambuf* oriBuff = nullptr;
std::stringstream errBuffer;
std::streambuf* oriErrBuff = nullptr;
}
class TEST_TASK_EXECUTOR : public testing::Test {
protected:
    void SetUp()
    {
        oriBuff = std::cout.rdbuf();
        std::cout.rdbuf(buffer.rdbuf());
        oriErrBuff = std::cerr.rdbuf();
        std::cerr.rdbuf(errBuffer.rdbuf());
    }

    void TearDown()
    {
        if (oriBuff != nullptr) {
            std::cout.rdbuf(oriBuff);
            std::cout<< buffer.str()<<std::endl;
            buffer.str("");
        }

        if (oriErrBuff != nullptr) {
            std::cerr.rdbuf(oriErrBuff);
            std::cerr<< errBuffer.str()<<std::endl;
            errBuffer.str("");
        }
        GlobalMockObject::verify();
    }
    // 静态成员：所有测试用例共享
    static std::string gpp_path;

    // 类级别初始化（整个测试类执行前运行一次）
    static void SetUpTestSuite() {
        gpp_path = FindGppPath();
        if (gpp_path.empty()) {
            FAIL() << "Failed to find g++ compiler";
        }
    }
    static std::string FindGppPath() {
        // 尝试使用 which 命令
        FILE* pipe = popen("which g++ 2>/dev/null", "r");
        if (pipe) {
            char buffer[128];
            if (fgets(buffer, sizeof(buffer), pipe)) {
                std::string path(buffer);
                // 移除换行符
                size_t pos = path.find_last_not_of("\n");
                if (pos != std::string::npos) {
                    path.erase(pos + 1);
                }
                pclose(pipe);
                if (access(path.c_str(), X_OK) == 0) {
                    return path;
                }
            }
            pclose(pipe);
        }
        return "";//没找到返回空字符串
    }

    inline void CheckAndRemoveFile(const std::string& filename) {
        ASSERT_TRUE(access(filename.c_str(), F_OK) == 0) << "file not exists:" << filename;
        ASSERT_TRUE(std::remove(filename.c_str()) == 0) << "remove file failed:" << filename;
    }
};
// 必须定义静态成员（关键）
std::string TEST_TASK_EXECUTOR::gpp_path;

TEST_F(TEST_TASK_EXECUTOR, task_executor_no_group)
{
    TaskExecutor exec;
    bool res = exec.ExecuteTasks();
    bool expect = true;
    EXPECT_EQ(res, expect);
}

TEST_F(TEST_TASK_EXECUTOR, task_executor_one_group_no_task)
{
    TaskExecutor exec;
    TaskGroup group_1(TaskStep::CODEGEN);
    exec.AddTaskGroup(group_1);

    bool res = exec.ExecuteTasks();
    bool expect = true;
    EXPECT_EQ(res, expect);
}

TEST_F(TEST_TASK_EXECUTOR, task_executor_one_group_one_task)
{
    std::string inputFile =
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/input_demo_1.cpp";
    std::string preprocessFile =
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/input_demo_1.ii";
    TaskExecutor exec;
    TaskGroup group_1(TaskStep::CODEGEN);
    group_1.AddTask(Task(gpp_path).AddArg("-E").AddArg(inputFile).AddArg("-o").AddArg(preprocessFile));
    exec.AddTaskGroup(group_1);

    bool res = exec.ExecuteTasks();
    bool expect = true;
    EXPECT_EQ(res, expect);
    // 验证文件是否存在，存在则删除
    CheckAndRemoveFile(preprocessFile);
}

TEST_F(TEST_TASK_EXECUTOR, task_executor_one_group_two_task)
{
    std::string inputFile =
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/input_demo_1.cpp";
    std::string preprocessFile =
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/input_demo_2.ii";
    std::string objectFile =
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/input_demo_2.o";
    TaskExecutor exec;
    TaskGroup group_1(TaskStep::CODEGEN);
    group_1.AddTask(Task(gpp_path).AddArg({"-E", inputFile, "-o", preprocessFile}));
    group_1.AddTask(Task(gpp_path).AddArg("-c").AddArg(preprocessFile).AddArg("-o").AddArg(objectFile));
    exec.AddTaskGroup(group_1);

    bool res = exec.ExecuteTasks();
    bool expect = true;
    EXPECT_EQ(res, expect);

    // 验证文件是否存在，存在则删除
    CheckAndRemoveFile(preprocessFile);
    CheckAndRemoveFile(objectFile);
}

TEST_F(TEST_TASK_EXECUTOR, task_executor_two_group_two_task)
{
    std::string inputFile_1 =
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/input_demo_1.cpp";
    std::string preprocessFile_1 =
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/input_demo_1.ii";
    TaskExecutor exec;
    TaskGroup group_1(TaskStep::CODEGEN);
    group_1.AddTask(Task(gpp_path).AddArg("-E").AddArg(inputFile_1).AddArg("-o").AddArg(preprocessFile_1));
    exec.AddTaskGroup(group_1);
    std::string inputFile_2 =
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/input_demo_2.cpp";
    std::string preprocessFile_2 =
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/input_demo_2.ii";
    TaskGroup group_2(TaskStep::CODEGEN);
    group_2.AddTask(Task(gpp_path).AddArg("-E").AddArg(inputFile_2).AddArg("-o").AddArg(preprocessFile_2));
    exec.AddTaskGroup(group_2);

    bool res = exec.ExecuteTasks();
    bool expect = true;
    EXPECT_EQ(res, expect);

    // 验证文件是否存在，存在则删除
    CheckAndRemoveFile(preprocessFile_1);
    CheckAndRemoveFile(preprocessFile_2);
}

TEST_F(TEST_TASK_EXECUTOR, task_executor_two_step_two_group_two_task)
{
    std::string inputFile_1 =
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/input_demo_1.cpp";
    std::string preprocessFile_1 =
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/input_demo_1.ii";
    TaskExecutor exec;
    TaskGroup group_1(TaskStep::CODEGEN);
    group_1.AddTask(Task(gpp_path).AddArg("-E").AddArg(inputFile_1).AddArg("-o").AddArg(preprocessFile_1));
    exec.AddTaskGroup(group_1);
    std::string inputFile_2 =
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/input_demo_2.cpp";
    std::string objectFile_2 =
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/input_demo_2.o";
    TaskGroup group_2(TaskStep::COMPILE);
    group_2.AddTask(Task(gpp_path).AddArg("-E").AddArg(inputFile_2).AddArg("-o").AddArg(objectFile_2));
    exec.AddTaskGroup(group_2);

    bool res = exec.ExecuteTasks();
    bool expect = true;
    EXPECT_EQ(res, expect);

    // 验证文件是否存在，存在则删除
    CheckAndRemoveFile(preprocessFile_1);
    CheckAndRemoveFile(objectFile_2);
}

TEST_F(TEST_TASK_EXECUTOR, task_executor_two_step_three_group_five_task)
{
    std::string inputFile_1 =
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/input_demo_1.cpp";
    std::string objectFile_1 =
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/input_demo_1.o";
    TaskExecutor exec;
    TaskGroup group_1(TaskStep::COMPILE);
    group_1.AddTask(Task(gpp_path).AddArg("-c").AddArg(inputFile_1).AddArg("-o").AddArg(objectFile_1));
    exec.AddTaskGroup(group_1);
    std::string inputFile_2 =
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/input_demo_2.cpp";
    std::string objectFile_2 =
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/input_demo_2.o";
    TaskGroup group_2(TaskStep::COMPILE);
    group_2.AddTask(Task(gpp_path).AddArg("-c").AddArg(inputFile_2).AddArg("-o").AddArg(objectFile_2));
    exec.AddTaskGroup(group_2);

    std::string binFile =
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/input_demo.bin";
    TaskGroup group_3(TaskStep::LINK);
    group_3.AddTask(Task(gpp_path).AddArg(objectFile_1).AddArg(objectFile_2).AddArg("-o").AddArg(binFile));
    exec.AddTaskGroup(group_3);

    bool res = exec.ExecuteTasks();
    bool expect = true;
    EXPECT_EQ(res, expect);

    // 验证文件是否存在，存在则删除
    CheckAndRemoveFile(objectFile_1);
    CheckAndRemoveFile(objectFile_2);
    CheckAndRemoveFile(binFile);
}

TEST_F(TEST_TASK_EXECUTOR, task_executor_one_group_two_task_launch_failed)
{
    std::string inputFile =
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/input_demo_1.cpp";
    std::string preprocessFile =
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/input_demo_1.ii";
    std::string objectFile =
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/input_demo_1.o";
    TaskExecutor exec;
    TaskGroup group_1(TaskStep::CODEGEN);
    group_1.AddTask(Task("error_exe").AddArg("-E").AddArg(inputFile).AddArg("-o").AddArg(preprocessFile));
    group_1.AddTask(Task(gpp_path).AddArg("-c").AddArg(preprocessFile).AddArg("-o").AddArg(objectFile));
    exec.AddTaskGroup(group_1);

    bool res = exec.ExecuteTasks();
    bool expect = false;
    EXPECT_EQ(res, expect);
}

TEST_F(TEST_TASK_EXECUTOR, task_executor_one_group_two_task_exec_failed)
{
    std::string inputFile =
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/input_demo_not_exist.cpp";
    std::string preprocessFile =
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/input_demo_1.ii";
    std::string objectFile =
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/input_demo_1.o";
    TaskExecutor exec;
    TaskGroup group_1(TaskStep::CODEGEN);
    group_1.AddTask(Task(gpp_path).AddArg("-E").AddArg(inputFile).AddArg("-o").AddArg(preprocessFile));
    group_1.AddTask(Task(gpp_path).AddArg("-c").AddArg(preprocessFile).AddArg("-o").AddArg(objectFile));
    exec.AddTaskGroup(group_1);

    bool res = exec.ExecuteTasks();
    bool expect = false;
    EXPECT_EQ(res, expect);
}

TEST_F(TEST_TASK_EXECUTOR, task_executor_one_group_two_task_exit_failed)
{
    std::string inputFile_1 =
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/input_demo_1.cpp";
    std::string objectFile_1 =
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/input_demo_1.o";
    TaskExecutor exec;
    TaskGroup group_1(TaskStep::COMPILE);
    group_1.AddTask(Task(gpp_path).AddArg("-c").AddArg(inputFile_1).AddArg("-o").AddArg(objectFile_1));
    exec.AddTaskGroup(group_1);
    std::string inputFile_3 =
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/input_demo_3.cpp";
    std::string objectFile_3 =
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/input_demo_3.o";
    TaskGroup group_3(TaskStep::COMPILE);
    group_3.AddTask(Task(gpp_path).AddArg("-c").AddArg(inputFile_3).AddArg("-o").AddArg(objectFile_3));
    exec.AddTaskGroup(group_3);

    std::string binFile =
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/input_demo.bin";
    TaskGroup group_4(TaskStep::DEVICE_LINK);
    group_4.AddTask(Task(gpp_path).AddArg(objectFile_1).AddArg(objectFile_3).AddArg("-o").AddArg(binFile));
    exec.AddTaskGroup(group_4);

    TaskGroup group_5(TaskStep::PACK_KERNEL);
    group_5.AddTask(Task(binFile));
    exec.AddTaskGroup(group_5);

    bool res = exec.ExecuteTasks();
    bool expect = false;
    EXPECT_EQ(res, expect);

    // 验证文件是否存在，存在则删除
    CheckAndRemoveFile(objectFile_1);
    CheckAndRemoveFile(objectFile_3);
    CheckAndRemoveFile(binFile);
}


TEST_F(TEST_TASK_EXECUTOR, task_executor_one_group_two_task_verbose)
{
    std::string inputFile =
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/input_demo_1.cpp";
    std::string preprocessFile =
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/input_demo_1.ii";
    std::string objectFile =
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/input_demo_1.o";
    TaskExecutor exec;
    TaskGroup group_1(TaskStep::CODEGEN);
    group_1.AddTask(Task(gpp_path).AddArg("-E").AddArg(inputFile).AddArg("-o").AddArg(preprocessFile));
    group_1.AddTask(Task(gpp_path).AddArg("-c").AddArg(preprocessFile).AddArg("-o").AddArg(objectFile));
    exec.AddTaskGroup(group_1);

    exec.PrintTasks();
    bool res = exec.ExecuteTasks();
    bool expect = true;
    EXPECT_EQ(res, expect);

    // 验证文件是否存在，存在则删除
    CheckAndRemoveFile(preprocessFile);
    CheckAndRemoveFile(objectFile);
}

TEST_F(TEST_TASK_EXECUTOR, task_executor_one_group_two_task_time)
{
    std::string inputFile =
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/input_demo_1.cpp";
    std::string preprocessFile =
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/input_demo_1.ii";
    std::string objectFile =
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/input_demo_1.o";
    TaskExecutor exec;
    TaskGroup group_1(TaskStep::CODEGEN);
    group_1.AddTask(Task(gpp_path).AddArg("-E").AddArg(inputFile).AddArg("-o").AddArg(preprocessFile));
    group_1.AddTask(Task(gpp_path).AddArg("-c").AddArg(preprocessFile).AddArg("-o").AddArg(objectFile));
    exec.AddTaskGroup(group_1);

    exec.EnableStatistics();
    bool res = exec.ExecuteTasks();
    bool expect = true;
    EXPECT_EQ(res, expect);
    // 验证文件是否存在，存在则删除
    CheckAndRemoveFile(preprocessFile);
    CheckAndRemoveFile(objectFile);
}

TEST_F(TEST_TASK_EXECUTOR, task_executor_one_group_two_task_verbose_time)
{
    std::string inputFile =
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/input_demo_1.cpp";
    std::string preprocessFile =
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/input_demo_1.ii";
    std::string objectFile =
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/input_demo_1.o";
    TaskExecutor exec;
    TaskGroup group_1(TaskStep::CODEGEN);
    group_1.AddTask(Task(gpp_path).AddArg("-E").AddArg(inputFile).AddArg("-o").AddArg(preprocessFile));
    group_1.AddTask(Task(gpp_path).AddArg("-c").AddArg(preprocessFile).AddArg("-o").AddArg(objectFile));
    exec.AddTaskGroup(group_1);

    exec.PrintTasks();
    exec.EnableStatistics();
    bool res = exec.ExecuteTasks();
    bool expect = true;
    EXPECT_EQ(res, expect);

    // 验证文件是否存在，存在则删除
    CheckAndRemoveFile(preprocessFile);
    CheckAndRemoveFile(objectFile);
}