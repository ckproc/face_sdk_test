#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <math.h>
#include <vector>
#include <string>
#include <map>
#include <iomanip>
#include <cmath>
//#include <cmath>
//#include "detect_embed_search_verification.hpp"
using namespace std;
using namespace cv;

typedef pair<float, int> P;
typedef std::map<string, int> Dict;
typedef Dict::const_iterator It;

bool cmp(const P &a, const P &b){
	return a.first > b.first;
}

bool isNaN(double x) { 
  return x != x;
}

struct search_item{
 int index;
 float img_similarity;
};

void getname(string &match_face){
	const size_t last_slash_idx = match_face.find_last_of("/");
	if (std::string::npos != last_slash_idx)
	{
		match_face.erase(0, last_slash_idx + 1);
		}
		const size_t period_idx = match_face.rfind('.');
		if (std::string::npos != period_idx)
	{
		match_face.erase(period_idx);
	}
	
}

int compare_face(cv::Mat db_embs, cv::Mat embed, int &match_idx, float &identification_score){
	vector<P> cdists;
	for(int i=0; i<db_embs.rows; ++i){
		double sumA=0;
		double sumB=0;
		double cosine = 0;
		for(int j=0; j<db_embs.cols; ++j){
			sumA+=db_embs.ptr<float>(i)[j]*db_embs.ptr<float>(i)[j];
			sumB+=embed.ptr<float>(0)[j]*embed.ptr<float>(0)[j];
			cosine+=db_embs.ptr<float>(i)[j]*embed.ptr<float>(0)[j];
			//cosine+=(db_embs.ptr<float>(i)[j]-embed.ptr<float>(0)[j])*(db_embs.ptr<float>(i)[j]-embed.ptr<float>(0)[j]);
		}
		sumA=sqrt(sumA);
		//cout<<"sumA:"<<sumA<<endl;
		//cout<<"sumB:"<<sumB<<endl;
		sumB=sqrt(sumB);
		cosine/=sumA*sumB;
		//cosine = sqrt(cosine);
		//cosine = 2.0/(1.0+exp(cosine));
		if(isNaN(cosine)){
			cdists.push_back(make_pair(0.0,i));
		}
		else{
			cdists.push_back(make_pair(cosine,i));
		}
		
	}
	sort(cdists.begin(),cdists.end(),cmp);
	
	match_idx = cdists[0].second;
	identification_score = cdists[0].first;
}


vector<string> &split(const string &str, char delim, vector<string> &elems) {
    istringstream iss(str);
    for (string item; getline(iss, item, delim); )
        elems.push_back(item);
    return elems;
}

void load_matchlist(string match_list,Dict &match_query){
	ifstream fin(match_list);
	string s;
	while(getline(fin,s)){
		vector<string> result;
		split(s,' ', result);
		match_query[result[0]]=atoi(result[1].c_str());
	}
}

std::vector<search_item>  face_retrieval(cv::Mat &query_array, cv::Mat &db, float threshold){
	int top_rank =5;
	vector<P> cdists;
	vector<search_item> result;
	for(int i=0; i<db.rows; ++i){
		double sumA=0;
		double sumB=0;
		double cosine = 0;
		for(int j=0; j<query_array.cols; ++j){
			sumA+=db.ptr<float>(i)[j]*db.ptr<float>(i)[j];
			sumB+=query_array.ptr<float>(0)[j]*query_array.ptr<float>(0)[j];
			cosine+=db.ptr<float>(i)[j]*query_array.ptr<float>(0)[j];
		}
		sumA=sqrt(sumA);
		sumB=sqrt(sumB);
		cosine/=sumA*sumB;
		if(isNaN(cosine)){
			cdists.push_back(make_pair(0.0,i));
		}
		else{
			cdists.push_back(make_pair(cosine,i));
		}
		//cdists.push_back(make_pair(cosine,i));
	}
	
	sort(cdists.begin(),cdists.end(),cmp);
	//match_idx = cdists[0].second;
	//identification_score = cdists[0].first;
	//for(int i=0; i<cdists.size();++i){
	 //    if(cdists[i].first>=threshold){
	//		 search_item temp;
	 //     	 temp.index=cdists[i].second;
	 //        temp.img_similarity=cdists[i].first;
	//		 result.push_back(temp);
	//	 }
	//	 else{
	//		 break;
	//	 }
	//}
	
	for(int k=0;k<top_rank;++k){	
		search_item temp;
		temp.index=cdists[k].second;
		temp.img_similarity=cdists[k].first;
		result.push_back(temp);
	}
	return result;
}

int main(int argc, const char *argv[]){
	
	std::string db_path = std::string(argv[1]);
	std::string query_path = std::string(argv[2]);
	std::string match_list = std::string(argv[3]);
	float threshold = atof(argv[4]);
	float write_threshold = threshold;
	FileStorage db;
	db.open(db_path,FileStorage::READ);
	cv::Mat db_embs;
	db["embed"]>>db_embs;
	vector<string> db_paths;
    db["path"]>>db_paths;
	
	FileStorage query;
	query.open(query_path,FileStorage::READ);
	cv::Mat query_embs;
	query["embed"]>>query_embs;
	vector<string> query_paths;
    query["path"]>>query_paths;
	//cout.setf(std::ios::left);
	//cout.setf(ios::fixed); 
   // cout.precision(8);
	float tp=0.0;
	float fp=0.0;
	float hit=0.0;
	int p=0;
	float tn=0.0;
	float rank1=0.0;
	int num=0;
	vector<float> accu;
	Dict match_query;
	load_matchlist(match_list, match_query);
	ofstream infile;
	ofstream infile1;
	ofstream infile2;
	ofstream infile3;
	infile.open("fpr_tpr_recall.txt",ios::trunc);
	infile1.open("fn.txt",ios::trunc);
	infile2.open("fp.txt",ios::trunc);
	infile3.open("recall.txt",ios::trunc);
	for(int k=0;k<100;++k){
		threshold=1-float(k)/100;
		cout<<threshold<<endl;
		tp=0.0;
		fp=0.0;
		tn=0.0;
		p=0.0;
		hit=0.0;
		num=0.0;
	for(int i=0;i<query_paths.size();++i){
		string query_nm=query_paths[i];
		getname(query_nm);
		string query_full=query_nm+".jpg";
		Mat emb(1,128,CV_32FC1,Scalar::all(0.0));
		query_embs.row(i).copyTo(emb);
		int match_idx;
		float identification_score;
		compare_face(db_embs, emb, match_idx, identification_score);
		vector<search_item> result = face_retrieval(emb, db_embs, threshold);
		if(threshold==write_threshold){
			for(int l=0;l<result.size();++l){
				string rankname = db_paths[result[l].index];
				float cosi = result[l].img_similarity;
				getname(rankname);
				infile3<<cosi<<" "<<query_nm<<" "<<rankname<<endl;
			}
			
		}
		string re_name=db_paths[match_idx];
		getname(re_name);
		if(match_query[query_full]==1){
			p+=1;
			for(int i =0;i<result.size();++i){
				string t_name = db_paths[result[i].index];
				getname(t_name);
				if(query_nm==t_name){
				    hit+=1;
				}
			}
		  
			if(query_nm==re_name){
				    rank1+=1;
			}
			if(identification_score>threshold){
				
				if(query_nm==re_name){
				    tp+=1;
			     }
				else {
					if(threshold==write_threshold)
					 infile1<<identification_score<<" "<<query_nm<<" "<<re_name<<endl;
				}
			}
			else{
				if(threshold==write_threshold)
      				infile1<<identification_score<<" "<<query_nm<<" "<<re_name<<endl;
			}
		}
		else{
			if(identification_score>threshold){
				fp+=1;
				if(threshold==write_threshold)
				 infile2<<identification_score<<" "<<query_nm<<" "<<re_name<<endl;
			}
			else{
				tn+=1;
			}
		}
		/*
		if(identification_score>threshold){
			
			getname(re_name);
			if(query_nm==re_name&&match_query[query_full]==1){
				tp+=1;
			}
			
			else if(match_query[query_full]==0){
				fp+=1;
				cout<<query_nm<<" "<<re_name<<" "<<identification_score<<endl;
				infile<<query_nm<<" "<<re_name<<" "<<identification_score<<endl;
				//cout<<query_nm<<endl;
			}
			
		}
		else if(match_query[query_full]==0){
			tn+=1;
		}
		else{
			//infile1<<query_nm<<" "<<re_name<<" "<<identification_score<<endl;
		}
		*/
		num+=1;
		
		
	}
        cout<<"threshold = "<<threshold<<endl;	
		cout<<"num = "<<num<<endl;
		cout<<"p = "<<p<<endl;
		cout<<"tp = "<<tp<<endl;
		cout<<"fp = "<<fp<<endl;
		cout<<"TPR = "<<tp/float(p)<<endl;
		cout<<"FPR = "<<fp/float(num-p)<<endl;
		float accuracy=(tp+tn)/num;
		cout<<"accu = "<<accuracy<<endl;
		cout<<"rank1 ="<<rank1/float(p)<<endl;
		accu.push_back(accuracy);
		//cout.width(8);
		  infile << std::setw(6)<< std::fixed << std::setprecision(4)<<float(threshold)<<"  \t"<<fp/float(num-p)<<"  \t"<<tp/float(p)<<"  \t"<<accuracy<<"  \t"<<hit/float(p)<<"  \t"<<tp/(tp+fp)<<endl;
	}
	infile.close();
	infile1.close();
	infile2.close();
	infile3.close();
	auto max_index=max_element(accu.begin(),accu.end());
	std::cout << "accuracy is " << *max_index<< " when thresh is " << 1-float(std::distance(accu.begin(), max_index))/100 << std::endl;
	return 0;
}
