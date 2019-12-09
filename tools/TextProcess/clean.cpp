/*
 * @author: visionzhong
 * @date: 2019-06-26
 * @func:
 */

#include<iostream>
#include<fstream>

#include "clean.h"
#include "utf8.h"

#include "log/taf_logger.h"

// 构造函数
Clean::Clean(){
}

// 初始化函数
void Clean::initialize(const std::string zh_file){
	std::string line;
	const std::string separation = "\t";
	std::ifstream infile;
    infile.open(zh_file.c_str());
	std::vector<std::string> seg_list;
	while (!infile.eof()){
        std::getline(infile, line);
		Clean::trim(line);
		Clean::split(line, seg_list, separation);
		if (seg_list.size() != 2) {
			continue;
		}
		// std::cout << seg_list[0] << "\t" << seg_list[1] << std::endl;
		complex2simple[seg_list[0]] = seg_list[1];
		seg_list.clear();
	}
	// std::unordered_map<std::string, std::string>::iterator it; 
	std::map<std::string, std::string>::iterator it; 
	// 为什么这里不行？
	// for (it=complex2simple.begin(); it<complex2simple.end(); ++it){
	// 	std::cout << it->first << "\t" << it->second << std::endl;
	// }
	std::cout << "MYINFO - complex2simple count:\t" 
        << complex2simple.size() << std::endl;
}


// 大写转小写

void Clean::trim(std::string& s) {
	if (!s.empty()){
	   	s.erase(s.find_last_not_of("\n")+1);
		s.erase(s.find_last_not_of("\r")+1);
	}
}


// 切分句子
void Clean::split(const std::string &line, std::vector<std::string>& letter,
		const std::string separation){
	size_t begin = 0, pos = 0;
	std::string token;
	while ((pos=line.find(separation, begin)) != std::string::npos) {
		if (pos>begin){
			token = line.substr(begin, pos-begin);
			letter.push_back(token);
		}
		begin = pos + separation.size();
	}
	if (pos > begin) {
		token = line.substr(begin, pos-begin);
	}
	letter.push_back(token);
}


// 将句子切分成单字符
std::vector<std::string> Clean::splitWordIntoLetters(const std::string& word) const {
	char* charWord = (char*)word.c_str();
	char* b = charWord;
	char* e = charWord + strlen(charWord);
	
	std::vector<std::string> letters;
	while (b != e) {
		int end = utf8::next(b, e);
		std::vector<unsigned char> utf8result;
		utf8::utf32to8(&end,&end + 1, std::back_inserter(utf8result));
		// letters.emplace_back(utf8result.begin(), utf8result.end());
        std::string result="";
        for (std::vector<unsigned char>::iterator it=utf8result.begin();
                it<utf8result.end(); ++it){
            result += *it;
        }
		letters.push_back(result);
	}
	return letters;
}


// 大写转小写
void Clean::toLower(std::string &str) {
	transform(str.begin(), str.end(), str.begin(), ::tolower);
}


// 全角转半角
std::string toFull(std::string str){
	std::string result = "";
	unsigned char tmp; unsigned char tmp1;
	for (unsigned int i = 0; i < str.length(); i++){
		tmp = str[i];
		tmp1 = str[i + 1];
		//cout << "uchar:" << (int) tmp << endl;
		if (tmp>32 && tmp<128){//是半角字符
			result += 163;//第一个字节设置为163
			result += (unsigned char)str[i]+128;//第二个字节+128;
		} else if (tmp >= 163){//是全角字符
			result += str.substr(i, 2);
			i++;
			continue;
		} else if (tmp == 32){//处理半角空格
			result += 161;
			result += 161;
		} else{
			result += str.substr(i, 2);
			i++;
		}
	}
	return result;
}


std::string Clean::toHalf(std::string input) {
	std::string temp;
	for (size_t i = 0; i < input.size(); i++) {
		if (((input[i] & 0xF0) ^ 0xE0) == 0) {
			int old_char = (input[i] & 0xF) << 12 
				| ((input[i + 1] & 0x3F) << 6 | (input[i + 2] & 0x3F));
			if (old_char == 0x3000) { // blank
				char new_char = 0x20;
				temp += new_char;
			} else if (old_char >= 0xFF01 && old_char <= 0xFF5E) { // full char
				char new_char = old_char - 0xFEE0;
				temp += new_char;
			} else { // other 3 bytes char
				temp += input[i];
				temp += input[i + 1];
				temp += input[i + 2];
			}
			i = i + 2;
		} else {
			temp += input[i];
		}
	}
	return temp;
}


// 繁体转简体
std::string Clean::toSimple(std::string &str) {
	// 将文字转按字符切分
	std::vector<std::string> letter = Clean::splitWordIntoLetters(str);
	// std::unordered_map<std::string, std::string>::iterator table_it; 
	std::map<std::string, std::string>::iterator table_it; 
	std::string new_str, unit;
	for (std::vector<std::string>::iterator it=letter.begin(); it<letter.end(); ++it) {
		unit = *it;
		table_it = complex2simple.find(*it);
		if (table_it != complex2simple.end()) {
			unit = table_it->second;
			// std::cout << "FIND:\t" << unit << std::endl;
		}
		new_str += unit;
	}
	return new_str;
}


// 丢弃特定字符串
void Clean::dropString(std::string &text, const std::string str){
    size_t str_size = str.size();
    if (str_size==0){
        return;
    }
    unsigned int begin = text.size()+1;
    while (begin != text.size()){
        // FDLOG("debug") << "begin: " << begin
        //     << "\ttext: " << text
        //     << "\ttext size: " << text.size()
        //     << "\tstr: " << str << endl;
        int pos = text.find(str);
        if (pos == -1){
            break;
        }
        begin = text.size();
        text.erase(pos, str_size);
    }
}


// 完整处理流程
std::string Clean::process(const std::string &str) {
    std::string new_str = str;
	Clean::toLower(new_str);
	new_str = Clean::toHalf(new_str);
    Clean::dropString(new_str, " ");  // 删除空格
	new_str = Clean::toSimple(new_str);
    return new_str;
}
