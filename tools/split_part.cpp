        int total_fid_num = (*_conf)["total_fid_num"].to_int32();
        int part_index = (*_conf)["part_index"].to_int32();
        /*part0输出预估打分的log*/
        if (part_index == 0) {
            int64_t uid = _msdata->input->uid;
            for (size_t i = 0; i < term_fids_dic.size() && i < total_fid_num; i++) {
                LOG(INFO) << "logid="     << _request->logid()
                          << " method="   << _request->method()
                          << " uid "      << uid
                          << " pred_idx " << i
                          << " term "     << term_fids_dic[i].second.fid
                          << " pred "     << term_fids_dic[i].first;
            }
        }
        total_part_num = (*_conf)["total_part_num"].to_int32();
        int real_fid_num = term_fids_dic.size();
        if (real_fid_num <= 0){
            LOG(INFO) << "logid=" << _request->logid() << "no fid real_fid_num:"<<real_fid_num;
            return 0;
        }
        if (real_fid_num < total_part_num){
            total_part_num = 1;
        }
        if (real_fid_num < total_fid_num){
            total_fid_num = real_fid_num;
        }
        int per_fid_num = total_fid_num / total_part_num;
        int remaind = total_fid_num - per_fid_num * total_part_num;
        int temp_size = per_fid_num;
        int temp_part_index = 0;
        int part_start_index = 0;
        int real_part_size = per_fid_num;
        if (part_index < remaind){
            real_part_size++;
        }

        for (int i=0; i<total_part_num; i++){
            int temp_part_size = per_fid_num;
            if (part_index == 0) {
                part_start_index = 0;
                break;
            }
            if (i < remaind){
                temp_part_size++;
            }
            part_start_index = part_start_index + temp_part_size;
            if (part_index == (i+1)){
                break;
            }
        }

        std::vector<std::string> redis_keys;
        std::map<std::string, std::string> redis_values;
        {
            int temp_index = 0;
            for (auto& item : term_fids_dic) {
                if (temp_index < part_start_index){
                    temp_index++;
                    continue;
                }
                redis_keys.push_back(base::Int64ToString(item.second.fid));
                LOG(INFO) << "hq_his_fid: " << item.second.fid;
                bad_fids_dic.insert(std::make_pair(item.second.fid, item.second));  // 记录处理过的吧
                if (redis_keys.size() >= real_part_size) {
                    LOG(INFO) << "logid=" << _request->logid()
                              << " method=" << _request->method()
                              <<" total_fid_num="<<total_fid_num
                              <<" part_start_index="<<part_start_index<<" real_part_size="<<real_part_size;
                    break;
                }
                temp_index++;
            }
            RpcWrapper rpc_wrapper;
            RedisForumYuelaou redis_forum_yuelaou;
            {
                TASK_TIMER_SCOPED(io);
                rpc_wrapper.set_log_id(_request->logid());
                rpc_wrapper.mget(redis_forum_yuelaou, "hot_thread_", redis_keys, redis_values, 10);
            }

            output_data.reserve(redis_values.size() * 500);
            {
                TASK_TIMER_SCOPED(iop);
                for (const auto& redis_value : redis_values) {
                    if (redis_value.second.empty()) {
                        continue;
                    }
                    int64_t fid = std::atol(redis_value.first.c_str());;
                    std::string fdir = "";
                    std::string sdir = "";
                    std::vector<std::string> dirs; 
                    if (HotCollection::forum_info->get(base::Uint64ToString(fid), dirs) == 0) {
                        if (dirs.size() >= 2) {
                            fdir = dirs[0];
                            sdir = dirs[1];
                        }
                    }
                    int item_index = 0;
                    for (base::StringSplitter item_ptr(redis_value.second.c_str(),';' ); item_ptr;
                         ++item_ptr) {
                        if (item_index >= 500) {
                            break;
                        }
                        SourceUnit ms_unit;
                        {
                            ms_unit.fid             = std::atol(redis_value.first.c_str());
                            ms_unit.source          = source;  // source
                            ms_unit.score.ctr_score = 0.0;
                            ms_unit.score.heat      = 0.0;
                            ms_unit.first_dir = fdir;  //add fdir by mxp
                            ms_unit.second_dir = sdir;    //add sdir by mxp
                        }
                        int field_index = 0;
                        for (base::StringSplitter field_ptr(item_ptr.field(),
                                                            item_ptr.field() + item_ptr.length(), ':');
                             field_ptr; ++field_ptr) {
                            if (field_index == 0) {
                                field_ptr.to_long(&(ms_unit.tid));
                            } else if (field_index == 1) {
                                field_ptr.to_double(&(ms_unit.score_quality));
                            } else if (field_index == 2) {
                                field_ptr.to_double(&(ms_unit.score.heat));
                            } else if (field_index == 3) {
                                ms_unit.type = std::string(field_ptr.field(), field_ptr.length());
                            }
                            ++field_index;
                        }
                        if (_is_thread_bad(ms_unit.tid, _msdata->input->bad_tids)) {
                            continue;
                        }
                        output_data.push_back(ms_unit);
                        ++item_index;
                    }
                }
            }
        }

