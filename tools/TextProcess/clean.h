/*
 * @author: visionzhong
 * @date: 2019-06-26
 * @func: 预处理通用函数
 */

#pragma once

#include<algorithm>
// #include<unordered_map>
#include<map>
#include<vector>
#include<string.h>


class Clean{

	private:
		std::map<std::string, std::string> complex2simple;
	
		// method
		
	public:
		Clean();
		void initialize(const std::string zh_file);
	   	// 大写字母转小写字母
		void toLower(std::string &str);
		// 全角转半角
		std::string toHalf(std::string str);
		// 繁体转简体
		std::string toSimple(std::string &str);
        // 将句子切分成单字符
		std::vector<std::string> splitWordIntoLetters(
                const std::string& word) const;
        // 对字符串进行按token切分
		void split(const std::string &line, std::vector<std::string> &seg_list,
				const std::string separation);
        // 清除句子结束符
		void trim(std::string &str);
        void dropString(std::string &text, const std::string str);
		// 完整预处理流程
        std::string process(const std::string &str);

};
