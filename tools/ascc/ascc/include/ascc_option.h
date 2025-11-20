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
 * \file ascc_option.h
 * \brief
 */
#ifndef __INCLUDE_ASCC_OPTION_H__
#define __INCLUDE_ASCC_OPTION_H__
#include <map>
#include <cstdint>
#include <vector>
#include <string>
#include <iostream>
#include <type_traits>
#include <functional>
#include "ascc_mlog.h"

namespace Ascc {
enum class ArgOccNumFlag : uint32_t {  // args occurence num
    OPTIONAL = 0x00,        // times: 0 / 1
    ZERO_OR_MORE = 0x01,    // times: >= 0
    REQUIRED = 0x02         // times: 1
};

enum class HasArgFlag : uint32_t {
    OPTIONAL = 0x01,
    REQUIRED = 0x02,
    NONE = 0x03
};

enum class FormatFlag : uint32_t {
    NORMAL = 0x00,      // Nothing special
    POSITIONAL = 0x01,  // Is a positional argument, no '-' required
    PREFIX = 0x02,      // Can this option directly prefix its value?
};

enum MiscFlags : uint32_t {  // Miscellaneous flags to adjust argument
    COMMAS = 0x01,           // Should this List split between commas?
    UNKNOWN_OPT = 0x02,      // Should this List all unknown options?
};

struct Option {
    virtual bool HandleOccurrence(size_t pos, const std::string& argName, const std::string& arg) = 0;
    virtual bool GetOptionValue(std::vector<std::string>& values) = 0;
    virtual bool AddOccurrence(size_t pos, const std::string& argName, const std::string& value, bool multiArg = false);
    virtual void PrintHelpStr();
    virtual void ClearUp() = 0;
    virtual bool Error(const std::string& message, const std::string& arg, std::ostream& os = std::cerr);
    bool shortArgFlag = false;
    bool longArgFlag = false;
    ArgOccNumFlag occNumFlag;
    FormatFlag formatFlag;
    HasArgFlag hasArgFlag;
    uint32_t miscFlags;
    uint32_t occNum;
    uint32_t position;
    std::string shortArgStr;
    std::string longArgStr;
    std::string helpStr;
    std::string valueStr;
    bool HasLongArgStr() const
    {
        return longArgFlag;
    }
    bool HasShortArgStr() const
    {
        return shortArgFlag;
    }

    void AddArgument();
    void SetShortArgStr(const std::string &value);
    void SetLongArgStr(const std::string &value);

    void SetOccNumFlag(const ArgOccNumFlag flag)
    {
        occNumFlag = flag;
    }
    void SetFormatFlag(const FormatFlag flag)
    {
        formatFlag = flag;
    }
    void SetHasArgFlag(const HasArgFlag flag)
    {
        hasArgFlag = flag;
    }
    void SetMiscFlags(const MiscFlags flags)
    {
        miscFlags |= flags;       // bitmap, thus use |=
    }
    void SetPosition(const uint32_t pos)
    {
        position = pos;
    }
    void SetDescription(const std::string &desc)
    {
        helpStr = desc;
    }

    void SetValueDesc(const std::string &desc)
    {
        valueStr = desc;
    }

    HasArgFlag GetHasArgFlag() const
    {
        return hasArgFlag;
    }
    ArgOccNumFlag GetOccNumFlag() const
    {
        return occNumFlag;
    }
    FormatFlag GetFormatFlag() const
    {
        return formatFlag;
    }

    uint32_t GetMiscFlags() const
    {
        return miscFlags;
    }
    uint32_t GetOccNum() const
    {
        return occNum;
    }

    explicit Option(ArgOccNumFlag flag)
        : occNumFlag(flag),
          formatFlag(FormatFlag::NORMAL),
          hasArgFlag(HasArgFlag::REQUIRED),
          miscFlags(0x00),
          occNum(0),
          position(0)
    {
    }
    virtual ~Option() = default;
};

template <typename DataType>
struct OptionValue;

template <typename DataType, bool IsClass>
struct OptionValueBase {
    using WrapperType = OptionValue<DataType>;
    bool HasValue() const
    {
        return false;
    }
    const DataType &GetValue() const {}
    template <typename DT>
    void SetValue(const DT & /* val */) const
    {
    }

protected:
    ~OptionValueBase() = default;
};

template <typename DataType>
struct OptionValueBase<DataType, false> {
    using WrapperType = DataType;
    bool valid_ = false;
    DataType value_;
    bool HasValue() const
    {
        return valid_;
    }
    const DataType &GetValue() const
    {
        return value_;
    }
    void SetValue(const DataType &value)
    {
        value_ = value;
        valid_ = true;
    }

protected:
    OptionValueBase() = default;
    OptionValueBase(const OptionValueBase &) = default;
    OptionValueBase &operator=(const OptionValueBase &) = default;
    ~OptionValueBase() = default;
};

template <typename DataType>
struct OptionValue final : public OptionValueBase<DataType, std::is_class_v<DataType>> {
    using WrapperType = OptionValue<DataType>;
    explicit OptionValue() = default;
    explicit OptionValue(const DataType &value)
    {
        this->SetValue(value);
    }
    template <typename DT>
    OptionValue<DataType> &operator=(const DT &value)
    {
        this->SetValue(value);
        return *this;
    }
};

template <typename DataType, bool ExternalStorage, bool IsClass>
class OptionStorage {
    DataType *location_ = nullptr;
    OptionValue<DataType> default_;

public:
    bool SetLocation(Option & /* opt */, DataType &data)
    {
        if (location_) {
            return false;
        }
        location_ = &data;
        default_ = data;
        return false;
    }

    template <typename T>
    void SetValue(T &value, bool initial = false)
    {
        *location_ = value;
        if (initial) {
            default_ = value;
        }
    }
};

template <class DataType>
class OptionStorage<DataType, false, true> : public DataType {
public:
    OptionValue<DataType> default_;

    template <class T>
    void SetValue(T &value, bool initial = false)
    {
        DataType::operator=(value);
        if (initial) {
            default_ = value;
        }
    }

    const DataType &GetValue() const
    {
        return *this;
    }

    const OptionValue<DataType> &GetDefault() const
    {
        return default_;
    }
};

template <class DataType>
class OptionStorage<DataType, false, false> {
public:
    DataType value_;
    OptionValue<DataType> default_;

    // Make sure we initialize the value with the default constructor for the type.
    OptionStorage() : value_(DataType()), default_() {}

    template <class T>
    void SetValue(const T &value, bool initial = false)
    {
        value_ = value;
        if (initial) {
            default_ = value;
        }
    }

    DataType &GetValue()
    {
        return value_;
    }
    DataType GetValue() const
    {
        return value_;
    }

    const OptionValue<DataType> &GetDefault() const
    {
        return default_;
    }

    explicit operator DataType() const
    {
        return GetValue();
    }

    // If the datatype is a pointer, support -> on it.
    DataType operator->() const
    {
        return value_;
    }
};

class ShortDesc {
public:
    std::string &desc_;
    explicit ShortDesc(std::string desc) : desc_(desc) {}
    void Apply(Option &opt) const
    {
        opt.SetShortArgStr(desc_);
    }
};

class HelpDesc {
public:
    std::string &desc_;
    explicit HelpDesc(std::string desc) : desc_(desc) {}
    void Apply(Option &opt) const
    {
        opt.SetDescription(desc_);
    }
};

class ValueDesc {
public:
    std::string &desc_;
    explicit ValueDesc(std::string desc) : desc_(desc) {}
    void Apply(Option &opt) const
    {
        opt.SetValueDesc(desc_);
    }
};

template <typename Ty>
struct Init {
    const Ty &init;
    explicit Init(const Ty &Val) : init(Val) {}
    template <typename Opt>
    void Apply(Opt &opt) const
    {
        opt.SetInitialValue(init);
    }
};

template <class Ty>
struct ListInitializer {
    std::vector<Ty> &inits;
    explicit ListInitializer(std::vector<Ty> &vals) : inits(vals) {}
    template <typename Opt>
    void Apply(Opt &opt) const
    {
        opt.SetInitialValues(inits);
    }
};

template <class Ty>
ListInitializer<Ty> ListInit(std::vector<Ty> &vals)
{
    return ListInitializer<Ty>(vals);
}

template <typename R, typename Arg>
struct Callback {
    std::function<R(Arg)> cb_;
    explicit Callback(std::function<R(Arg)> cb) : cb_(cb) {}
    template <typename Opt>
    void Apply(Opt &opt) const
    {
        opt.SetCallback(cb_);
    }
};

template <typename Ty>
struct LocationClass {
    Ty &loc;
    explicit LocationClass(Ty &L) : loc(L) {}
    template <typename Opt>
    void Apply(Opt &opt) const
    {
        opt.SetLocation(opt, loc);
    }
};

template <typename Ty>
LocationClass<Ty> Location(Ty &L)
{
    return LocationClass<Ty>(L);
}

template <typename Arg>
struct Applicator {
    template <typename Opt>
    static void Apply(const Arg &arg, Opt &opt)
    {
        arg.Apply(opt);
    }
};

template <size_t n>
struct Applicator<char[n]> {
    template <typename Opt>
    static void Apply(std::string arg, Opt &opt)
    {
        opt.SetLongArgStr(arg);
    }
};

template <size_t n>
struct Applicator<const char[n]> {
    template <typename Opt>
    static void Apply(std::string arg, Opt &opt)
    {
        opt.SetLongArgStr(arg);
    }
};

template <>
struct Applicator<std::string> {
    template <typename Opt>
    static void Apply(const std::string &arg, Opt &opt)
    {
        opt.SetLongArgStr(arg);
    }
};

template <>
struct Applicator<MiscFlags> {
    static void Apply(MiscFlags miscFlags, Option &opt)
    {
        opt.SetMiscFlags(miscFlags);
    }
};

template <>
struct Applicator<ArgOccNumFlag> {
    static void Apply(ArgOccNumFlag occ, Option &opt)
    {
        opt.SetOccNumFlag(occ);
    }
};

template <>
struct Applicator<HasArgFlag> {
    static void Apply(HasArgFlag val, Option &opt)
    {
        opt.SetHasArgFlag(val);
    }
};

template <>
struct Applicator<FormatFlag> {
    static void Apply(FormatFlag format, Option &opt)
    {
        opt.SetFormatFlag(format);
    }
};

template <typename Opt, typename Arg, typename... Args>
void Apply(Opt *opt, const Arg &arg, const Args &...args)
{
    Applicator<Arg>::Apply(arg, *opt);
    Apply(opt, args...);
}

template <typename Opt, typename Arg, class... Args>
void Apply(Opt *opt, const Arg &arg)
{
    Applicator<Arg>::Apply(arg, *opt);
}

template <typename DataType>
class PaserTrait {
public:
    using TraitType = DataType;
};

template <typename DataType>
class OptionParser : public PaserTrait<DataType> {
public:
    bool Parse(Option &opt, const std::string &argName, const std::string &arg, DataType & /* val */) const
    {
        std::string argVal;
        if (opt.HasLongArgStr() || opt.HasShortArgStr()) {
            argVal = arg;
        } else {
            argVal = argName;
        }
        return opt.Error("Can't support parse this arg '" + argVal + "!", argName);
    }
};

template <>
class OptionParser<std::string> : public PaserTrait<std::string> {
public:
    bool Parse(Option & /* opt */, const std::string & /* argName */, const std::string &arg, std::string &val) const
    {
        val = arg;
        return false;
    }
};

template <>
class OptionParser<bool> : public PaserTrait<bool> {
public:
    bool Parse(Option &opt, const std::string &argName, const std::string &arg, bool &val) const
    {
        if (arg == "true" || arg == "TRUE" || arg == "True" || arg == "1" || arg == "") {
            val = true;
            return false;
        }

        if (arg == "false" || arg == "FALSE" || arg == "False" || arg == "0") {
            val = false;
            return false;
        }
        return opt.Error("'" + arg + "' is invalid value for bool argument! Try false or true", argName);
    }
};

template <>
class OptionParser<char> : public PaserTrait<char> {
public:
    bool Parse(Option & /* opt */, const std::string & /* argName */, const std::string &arg, char &val) const
    {
        val = arg[0];
        return false;
    }
};

template <>
class OptionParser<uint32_t> : public PaserTrait<uint32_t> {
public:
    bool Parse(Option &opt, const std::string &argName, const std::string &arg, uint32_t &val) const
    {
        try {
            val = std::stoul(arg);
        } catch (std::exception &ex) {
            return opt.Error("'" + arg + "' is invalid value for unsigned integer argument! ", argName);
        }

        return false;
    }
};

template <>
class OptionParser<int32_t> : public PaserTrait<int32_t> {
public:
    bool Parse(Option &opt, const std::string &argName, const std::string &arg, int32_t &val) const
    {
        try {
            val = std::stoi(arg);
        } catch (std::exception &ex) {
            return opt.Error("'" + arg + "' is invalid value for integer argument! ", argName);
        }
        return false;
    }
};

template <typename DataType, bool ExternalStorage = false, typename ParserPolicy = OptionParser<DataType>>
class Opt
    : public Option
    , public OptionStorage<DataType, ExternalStorage, std::is_class_v<DataType>> {
    ParserPolicy parser_;
    std::string oriValueStr_;
public:
    bool HandleOccurrence(size_t pos, const std::string &argName, const std::string &arg) override
    {
        typename ParserPolicy::TraitType val = typename ParserPolicy::TraitType();
        if (parser_.Parse(*this, argName, arg, val)) {
            return true;  // Parse error!
        }
        oriValueStr_ = arg;
        this->SetValue(val);
        this->SetPosition(pos);
        callback_(val);
        ASCC_LOGI("Opt '%s' HandleOccurrence %s", argName.c_str(), arg.c_str());
        return false;
    }

    // true means find occurence, false means not parsed before
    bool GetOptionValue(std::vector<std::string> &values) override
    {
        if (this->occNum == 0) {
            return false;
        }
        values.push_back(this->oriValueStr_);
        return true;
    }

    void ClearUp() override
    {
        this->occNum = 0;
        this->oriValueStr_.clear();
    }

    void Done()
    {
        AddArgument();
    }
    template <typename... Args>
    explicit Opt(const Args &...arg) : Option(ArgOccNumFlag::OPTIONAL)
    {
        Apply(this, arg...);
        Done();
    }
    void SetInitialValue(const DataType &value)
    {
        this->SetValue(value, true);
    }
    void SetCallback(std::function<void(const typename ParserPolicy::TraitType &)> cb)
    {
        callback_ = cb;
    }

    std::function<void(const typename ParserPolicy::TraitType &)> callback_ =
        [](const typename ParserPolicy::TraitType &) {
        };
};

template <typename DataType, typename StorageClass>
class OptListStorage {
    StorageClass *location_ = nullptr;
    std::vector<OptionValue<DataType>> default_ = std::vector<OptionValue<DataType>>();

public:
    OptListStorage() = default;
    void Clear() const {}
    bool SetLocation(Option & /* opt */, StorageClass &location)
    {
        if (location) {
            return true;
        }
        location_ = &location;
        return false;
    }

    template <typename T>
    void AddValue(const T &value, bool initial = false)
    {
        location_->push_back(value);
        if (initial) {
            default_.push_back(value);
        }
    }

    const std::vector<OptionValue<DataType>> &GetDefault() const
    {
        return default_;
    }
};

template <typename DataType>
class OptListStorage<DataType, bool> {
    std::vector<DataType> storage_;
    std::vector<OptionValue<DataType>> default_;

public:
    using iterator = typename std::vector<DataType>::iterator;
    iterator begin()
    {
        return storage_.begin();
    }
    iterator end()
    {
        return storage_.end();
    }

    using SizeType = typename std::vector<DataType>::size_type;
    SizeType size() const
    {
        return storage_.size();
    }
    bool empty() const
    {
        return storage_.empty();
    }
    void clear()
    {
        storage_.clear();
    }

    std::vector<DataType> *operator&()
    {
        return &storage_;
    }
    const std::vector<DataType> *operator&() const
    {
        return &storage_;
    }

    template <typename T>
    void AddValue(const T &value, bool initial = false)
    {
        storage_.push_back(value);
        if (initial) {
            default_.push_back(OptionValue<DataType>(value));
        }
    }

    const std::vector<OptionValue<DataType>> &GetDefault() const
    {
        return default_;
    }
};

template <typename DataType, typename StorageClass = bool, typename ParserPolicy = OptionParser<DataType>>
class OptList
    : public Option
    , public OptListStorage<DataType, StorageClass> {
    std::vector<uint32_t> positions_;
    ParserPolicy parser_;
    std::vector<std::string> orgValues_;
public:
    bool HandleOccurrence(size_t pos, const std::string &argName, const std::string &arg) override
    {
        typename ParserPolicy::TraitType val = typename ParserPolicy::TraitType();
        if (parser_.Parse(*this, argName, arg, val)) {
            return true;  // Parse error!
        }
        OptListStorage<DataType, StorageClass>::AddValue(val);
        this->SetPosition(pos);
        positions_.push_back(pos);
        orgValues_.push_back(arg);
        callback_(val);
        ASCC_LOGI("OpList '%s' HandleOccurrence %s", argName.c_str(), arg.c_str());
        return false;
    }

    // true means find occurence, false means not parsed before
    bool GetOptionValue(std::vector<std::string> &values) override
    {
        if (this->occNum == 0) {
            return false;
        }
        values = orgValues_;
        return true;
    }
    void Done()
    {
        AddArgument();
    }

    void ClearUp() override
    {
        OptListStorage<DataType, StorageClass>::clear();
        this->orgValues_.clear();
        this->positions_.clear();
        this->occNum = 0;
    }

    template <typename... Mods>
    explicit OptList(const Mods &...Ms) : Option(ArgOccNumFlag::ZERO_OR_MORE)
    {
        Apply(this, Ms...);
        Done();
    }
    void SetInitialValues(const std::vector<DataType> &value)
    {
        for (auto &val : value) {
            OptListStorage<DataType, StorageClass>::AddValue(val, true);
        }
    }

    void SetCallback(std::function<void(const typename ParserPolicy::TraitType &)> cb)
    {
        callback_ = cb;
    }

    std::function<void(const typename ParserPolicy::TraitType &)> callback_ =
        [](const typename ParserPolicy::TraitType &) {
        };
};
void PrinterExit();
bool ParseCommandLineOptions(int32_t argc, const char *const argv[]);
AsccStatus GetPositionalInputFiles(std::vector<std::string> &files);
bool GetOptionValuesByArgName(const std::string &argName, std::vector<std::string> &argValues);
void ClearUpOptionValue();
void PrintRequireOptInfo(Ascc::Option* optPtr);
}
#endif  // COMMAND_LINE_H
