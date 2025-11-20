/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file ascc_option.cpp
 * \brief
 */
#include "ascc_option.h"

#include <map>
#include <vector>
#include <iomanip>

#include "ascc_utils.h"

namespace Ascc {
static inline std::ostream &Outs()
{
    return std::cout;
}

template <typename T>
class ManagerCommandOptions {
public:
    ManagerCommandOptions() = default;
    ~ManagerCommandOptions() = default;
    T *operator->() const
    {
        static T instance;
        return &instance;
    }
};

class CommandOptionParser {
    std::vector<Option *> inputFileOpts_;
    std::vector<std::string> argsOrder_;
    std::map<std::string, Option *> argOptMap_;
    std::string programVersion_ = "ASCC Version 1.0.0";

public:
    std::string programName_;
    CommandOptionParser()
    {
        argOptMap_.clear();
        argsOrder_.clear();
        inputFileOpts_.clear();
    }

    const std::string &GetVersionMessage() const
    {
        return programVersion_;
    }
    bool ParserCommandOptions(const std::vector<std::string> &argv, std::ostream &os = std::cerr);
    bool GetOptionValuesByArgName(const std::string &argName, std::vector<std::string> &argValues)
    {
        auto iter = argOptMap_.find(argName);
        if (iter == argOptMap_.end()) {
            return false;
        }
        return iter->second->GetOptionValue(argValues);
    }

    // Update input files into inputOpt
    AsccStatus GetPositionalInputFiles(std::vector<std::string> &inputOpt)
    {
        for (const auto &fileOpt : inputFileOpts_) {
            ASCC_CHECK((fileOpt->GetOptionValue(inputOpt)),
                {ASCC_LOGE("Extract filename from GetOptionValue failed.");});
        }
        for (const auto &fileName : inputOpt) {
            ASCC_LOGI("Input filename from GetPositionalInputFiles: %s.", fileName.c_str());
        }
        return AsccStatus::SUCCESS;
    }

    void ClearUp()
    {
        for (const auto &fileOpt : this->inputFileOpts_) {
            fileOpt->ClearUp();
        }
        for (const auto &opt : this->argOptMap_) {
            opt.second->ClearUp();
        }
    }

    void AddOption(Option *opt)
    {
        if (opt == nullptr) {
            ASCC_LOGE("Option Initial failed.");
        }
        if (argOptMap_.count(opt->longArgStr) > 0 || argOptMap_.count(opt->shortArgStr) > 0) {
            ASCC_LOGI("Option is invalid or add repeated option long : '%s', short : '%s'", opt->longArgStr.c_str(),
                      opt->shortArgStr.c_str());
            return;
        }

        if (opt->HasLongArgStr() && argOptMap_.count(opt->longArgStr) == 0) {
            argOptMap_.insert(std::make_pair(opt->longArgStr, opt));
        }

        if (opt->HasShortArgStr() && argOptMap_.count(opt->shortArgStr) == 0) {
            argOptMap_.insert(std::make_pair(opt->shortArgStr, opt));
        }

        // save short/long option arg by define order
        if (opt->HasShortArgStr() && !opt->HasLongArgStr()) {
            argsOrder_.push_back(opt->shortArgStr);
        } else {
            argsOrder_.push_back(opt->longArgStr);
        }

        if (opt->GetFormatFlag() == FormatFlag::POSITIONAL) {
            ASCC_LOGI("Add position option '%s'.", opt->valueStr.c_str());
            inputFileOpts_.push_back(opt);
        }
    }

    void PrintOption(Option* opt) const
    {
        opt->PrintHelpStr();
    }

    void PrintAllOption()
    {
        for (const auto& arg : argsOrder_) {
            auto iter = argOptMap_.find(arg);
            if (iter == argOptMap_.end()) {
                continue;
            }
            PrintOption(iter->second);
        }
    }

private:
    bool HasArg(const Option *opt) const
    {
        return opt->GetOccNumFlag() == ArgOccNumFlag::REQUIRED;
    }
    Option *LookupOption(std::string &arg, std::string &value);
    Option *LookupLongOption(std::string &arg, std::string &value);
    Option *FindPrefixOption(std::string arg, size_t &length);
    Option *LookupPrefixOption(std::string &arg, std::string &value);
    void LookupNearestOption(std::string & /* arg */) const {}
    bool ProvideOption(Option *handler, const std::string &argName, std::string &value, size_t argc,
                       const std::vector<std::string> &argv, size_t &idx) const;
    bool ParserPositionalOptions(std::vector<std::pair<std::string, size_t>> &positionalVals);
    bool ProvidePositionOption(Option *handler, std::string &value, size_t idx) const;
    bool CommaSeparateAndAddOccurrence(Option *handler, const std::string &argName, std::string &value, size_t idx,
                                       bool multiArg = false) const;
};

Option *CommandOptionParser::LookupOption(std::string &arg, std::string &value)
{
    if (arg.empty()) {
        ASCC_LOGE("Lookup option failed, arg is empty.");
        return nullptr;
    }

    size_t equalPos = arg.find('=');
    if (equalPos == std::string::npos) {
        ASCC_LOGI("Can't find equal of value, arg is '%s'.", arg.c_str());
    } else {
        ASCC_LOGI("Find equal of value, arg is '%s' , option is '%s'.", arg.c_str(), arg.substr(0, equalPos).c_str());
    }
    auto iter = argOptMap_.find(arg.substr(0, equalPos));
    if (iter == argOptMap_.end()) {
        // may not be the type to be parsed in this method. Thus use LOGI instead of LOGE
        ASCC_LOGI("Can't lookup option '%s'.", arg.c_str());
        return nullptr;
    }

    if (equalPos != std::string::npos) {
        value = arg.substr(equalPos + 1);
        arg = arg.substr(0, equalPos);
    }

    return iter->second;
}

Option *CommandOptionParser::LookupLongOption(std::string &arg, std::string &value)
{
    ASCC_LOGI("Lookup long option '%s'.", arg.c_str());
    Option *opt = LookupOption(arg, value);
    if (opt == nullptr) {
        // may not be the type to be parsed in this method. Thus use LOGI instead of LOGE
        ASCC_LOGI("Can't lookup long option '%s'.", arg.c_str());
        return nullptr;
    }
    ASCC_LOGI("Lookup option '%s', value is '%s' success.", arg.c_str(), value.c_str());
    return opt;
}

Option *CommandOptionParser::FindPrefixOption(std::string arg, size_t &length)
{
    if (arg.size() <= 1) {
        return nullptr;
    }

    auto opt = argOptMap_.find(arg);
    if (opt != argOptMap_.end() && opt->second->GetFormatFlag() != FormatFlag::PREFIX) {
        opt = argOptMap_.end();
    }

    while (opt == argOptMap_.end() && arg.size() > 1) {
        arg = arg.substr(0, arg.size() - 1);
        opt = argOptMap_.find(arg);
        if (opt != argOptMap_.end() && opt->second->GetFormatFlag() != FormatFlag::PREFIX) {
            opt = argOptMap_.end();
        }
    }
    if (opt != argOptMap_.end() && opt->second->GetFormatFlag() == FormatFlag::PREFIX) {
        length = arg.size();
        return opt->second;
    }

    return nullptr;
}

Option *CommandOptionParser::LookupPrefixOption(std::string &arg, std::string &value)
{
    ASCC_LOGI("Lookup prefix option '%s'.", arg.c_str());
    size_t argLength = 0;
    auto opt = FindPrefixOption(arg, argLength);
    if (opt == nullptr) {
        // may not be the type to be parsed in this method. Thus use LOGI instead of LOGE
        ASCC_LOGI("Can't lookup prefix option '%s'.", arg.c_str());
        return nullptr;
    }

    auto prefixValue = (argLength < arg.size()) ? arg.substr(argLength) : "";
    arg = arg.substr(0, argLength);
    if (prefixValue.empty() || (opt->GetFormatFlag() == FormatFlag::PREFIX && prefixValue[0] != '=')) {
        value = prefixValue;
        ASCC_LOGI("Lookup prefix option '%s' without '=', value is '%s' success.", arg.c_str(), value.c_str());
        return opt;
    }

    ASCC_LOGI("Can't lookup prefix option '%s'.", arg.c_str());
    return nullptr;
}

bool CommandOptionParser::ProvideOption(Option *handler, const std::string &argName, std::string &value, size_t argc,
                                        const std::vector<std::string> &argv, size_t &idx) const
{
    switch (handler->GetHasArgFlag()) {
        case HasArgFlag::REQUIRED:
            if (value.empty()) {  // no value?
                if (idx + 1 >= argc || argv[idx + 1][0] == '-') {
                    Ascc::HandleErrorAndCheckLog("Option " + argName + " requires a value!");
                    return handler->Error("requires a value!", argName);
                }

                value = argv[++idx];
                ASCC_LOGI("Option '%s' requires a value, value is '%s'.", argName.c_str(), value.c_str());
            }
            break;
        case HasArgFlag::OPTIONAL:
            ASCC_LOGI("Option '%s' optional.", argName.c_str());
            break;
        case HasArgFlag::NONE:
            if (!value.empty()) {
                Ascc::HandleErrorAndCheckLog("Option " + argName + " does not take a value!");
                return true;
            }
            break;
        default:
            break;
    }

    return CommaSeparateAndAddOccurrence(handler, argName, value, idx);
}

bool CommandOptionParser::ProvidePositionOption(Option *handler, std::string &value, size_t idx) const
{
    return ProvideOption(handler, handler->longArgStr, value, 0, std::vector<std::string>(), idx);
}

bool CommandOptionParser::CommaSeparateAndAddOccurrence(Option *handler, const std::string &argName, std::string &value,
                                                        size_t idx, bool multiArg) const
{
    // Check to see if this option accepts a comma separated list of values.  If
    // it does, we have to split up the value into multiple values.
    if ((handler->GetMiscFlags() & COMMAS) > 0) {
        std::string val(value);
        std::string::size_type pos = val.find(',');
        while (pos != std::string::npos) {
            // Process the portion before the comma.
            if (handler->AddOccurrence(idx, argName, val.substr(0, pos), multiArg)) {
                return true;
            }
            // Erase the portion before the comma, AND the comma.
            val = val.substr(pos + 1);
            // Check for another comma.
            pos = val.find(',');
        }
        value = val;
    }

    return handler->AddOccurrence(idx, argName, value, multiArg);
}

void PrintRequireOptInfo(Option* optPtr)
{
    std::string shortStr = optPtr->shortArgStr;
    std::string longStr = optPtr->longArgStr;
    if ((!shortStr.empty()) && (!longStr.empty())) {
        Ascc::HandleErrorAndCheckLog("Option --" + longStr + " / -" + shortStr + " is required.");
        return;
    } else if ((!shortStr.empty())) {
        Ascc::HandleErrorAndCheckLog("Option -" + shortStr + " is required.");
        return;
    } else if ((!longStr.empty())) {
        Ascc::HandleErrorAndCheckLog("Option --" + longStr + " is required.");
        return;
    }
}

bool CommandOptionParser::ParserPositionalOptions(std::vector<std::pair<std::string, size_t>> &positionalVals)
{
    bool err = false;
    if (inputFileOpts_.size() == 1) {
        ASCC_LOGI("Begin to parsing position option.");
        for (auto &val : positionalVals) {
            err = err || ProvidePositionOption(inputFileOpts_.front(), val.first, val.second);
        }
    } else {
        Ascc::HandleErrorAndCheckLog("Position option is too much.");
        err = true;
    }
    for (const auto &opt : argOptMap_) {
        switch (opt.second->GetOccNumFlag()) {
            case ArgOccNumFlag::REQUIRED:
                if (opt.second->GetOccNum() == 0) {
                    PrintRequireOptInfo(opt.second);
                    err = true;
                }
                [[fallthrough]];
            default:
                break;
        }
        if (err) {
            break;
        }
    }
    return err;
}

bool CommandOptionParser::ParserCommandOptions(const std::vector<std::string> &argv, std::ostream &os)
{
    programName_ = argv[0];
    std::vector<std::pair<std::string, size_t>> positionalVals;
    uint32_t firstArg = 1;
    bool err = false;
    for (size_t i = firstArg; i < argv.size(); ++i) {
        Option *handler = nullptr;
        std::string value;
        std::string argName = "";
        bool haveDoubleDash = false;
        if (argv[i][0] != '-') {  // find positional arg
            if (!inputFileOpts_.empty()) {
                positionalVals.push_back(std::make_pair(argv[i], i));
                continue;
            }
        } else {
            argName = argv[i].substr(1);
            if (argName[0] == '-') {
                haveDoubleDash = true;
                argName = argName.substr(1);
            }

            handler = LookupLongOption(argName, value);
            if (!handler && !haveDoubleDash) {
                handler = LookupPrefixOption(argName, value);
            }

            if (!handler) {
                LookupNearestOption(argName);
            }
        }

        if (!handler) {
            os << programName_ << ": Unknown command argument '" << argName << "'. Try: '" << programName_
               << " --help'\n";
            err = true;
            continue;
        }

        if (handler->GetFormatFlag() == FormatFlag::POSITIONAL) {
            handler->Error("This argument does not take a value.", argName);
        } else {
            err = err || ProvideOption(handler, argName, value, argv.size(), argv, i);
        }

        if (err) {
            Ascc::HandleErrorAndCheckLog("Parser " + argName + " command options failed.");
            return !err;
        }
    }

    err = err || ParserPositionalOptions(positionalVals);
    return !err;
}

static ManagerCommandOptions<CommandOptionParser> g_managerCommandOptions{};
bool Option::AddOccurrence(size_t pos, const std::string &argName, const std::string &value, bool multiArg)
{
    if (!multiArg) {
        ASCC_LOGI("Option '%s' add occurrence '%s' at pos '%zu'.", argName.c_str(), value.c_str(), pos);
        occNum++;  // Increment the number of times we have been seen
    }

    return HandleOccurrence(pos, argName, value);
}

void Option::PrintHelpStr()
{
    auto show = [this](bool longPrefix, bool withShort = false) {
        std::string argStr = this->shortArgStr;
        if (longPrefix) {
            argStr = this->longArgStr;
            Outs() << "-";
        }

        Outs() << "-" << argStr;
        if (!this->valueStr.empty()) {
            Outs() << "=" << this->valueStr;
        }

        if (withShort) {
            const size_t commnadWidth = 48;
            Outs() << std::setw(commnadWidth - argStr.size() -
                                (this->valueStr.size() == 0 ? 0 : (this->valueStr.size() + 1)));
            Outs() << "(-" << this->shortArgStr << ")";
        }
        Outs() << std::endl;
    };
    show(HasLongArgStr(), (HasLongArgStr() && HasShortArgStr()));

    size_t start = 0;
    size_t end = 0;
    while ((end = this->helpStr.find('\n', start)) != std::string::npos) {
        auto info = helpStr.substr(start, end - start);
        start = end + 1;
        if (info.empty()) {
            continue;
        }
        Outs() << "        " << info << std::endl;
    }

    if (start < this->helpStr.size()) {
        Outs() << "        " << helpStr.substr(start) << std::endl;
    }
    Outs() << std::endl;
}

bool Option::Error(const std::string &message, const std::string &arg, std::ostream &os)
{
    if (arg.empty()) {
        os << helpStr;
    } else {
        os << g_managerCommandOptions->programName_ << ": for the " << arg;
    }

    os << " option: " << message << "\n";
    return true;
}

void Option::AddArgument()
{
    g_managerCommandOptions->AddOption(this);
}

void Option::SetShortArgStr(const std::string &value)
{
    shortArgFlag = true;
    shortArgStr = value;

    if (!longArgFlag) {
        longArgStr = value;
    }
};

void Option::SetLongArgStr(const std::string &value)
{
    longArgFlag = true;
    longArgStr = value;

    if (!shortArgFlag) {
        shortArgStr = value;
    }
};

void PrinterExit()
{
    _Exit(0);
}

class VersionPrinter {
    const bool showHidden_;

public:
    explicit VersionPrinter(bool showHidden) : showHidden_(showHidden) {}
    virtual ~VersionPrinter() = default;
    void PrintVersion() const
    {
        Outs() << g_managerCommandOptions->GetVersionMessage() << "\n";
    }
    // Invoke the printer.
    void operator=(bool value)
    {
        if (!value) {
            return;
        }
        PrintVersion();
        PrinterExit();  // Halt the program since version information was printed
    }
};

class HelpPrinter {
    const bool showHidden_;

public:
    explicit HelpPrinter(bool showHidden) : showHidden_(showHidden) {}
    virtual ~HelpPrinter() = default;
    void PrintHelp() const
    {
        Outs() << "\nUsage   : bishengcc [options] file..."
               << "\n";
        Outs() << "\nOptions :\n";
        Outs() << "=======================================================\n";

        g_managerCommandOptions->PrintAllOption();
    }
    // Invoke the printer.
    void operator=(bool value)
    {
        if (!value) {
            return;
        }
        PrintHelp();
        PrinterExit();  // Halt the program since help information was printed
    }
};

namespace {
HelpPrinter printer{ false };
static Opt<HelpPrinter, true, OptionParser<bool>> g_hOpt("help", ShortDesc("h"),
                                                         HelpDesc("Print this help information on this tool."),
                                                         HasArgFlag::NONE, Location(printer));
}

bool ParseCommandLineOptions(int32_t argc, const char *const argv[])
{
    std::vector<std::string> newArgv;
    for (int32_t i = 0; i < argc; ++i) {
        newArgv.push_back(argv[i]);
    }

    return g_managerCommandOptions->ParserCommandOptions(newArgv);
}

AsccStatus GetPositionalInputFiles(std::vector<std::string>& files)
{
    return g_managerCommandOptions->GetPositionalInputFiles(files);
}

bool GetOptionValuesByArgName(const std::string &argName, std::vector<std::string>& argValues)
{
    return g_managerCommandOptions->GetOptionValuesByArgName(argName, argValues);
}

void ClearUpOptionValue() {
    g_managerCommandOptions->ClearUp();
}
}
