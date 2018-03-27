#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <string>

using namespace std;

vector<string> &split(const string &str, char delim, vector<string> &elems) {
    istringstream iss(str);
    for (string item; getline(iss, item, delim); )
        elems.push_back(item);
    return elems;
}

void load_matchlist(string match_list,vector<vector<string>> &match_query){
	ifstream fin(match_list);
	string s;
	while(getline(fin,s)){
		vector<string> result;
		split(s,' ', result);
		match_query.push_back(result);//[result[0]]=atoi(result[1].c_str());
	}
}

int main(int argc, const char *argv[]){
	std::string match_list = std::string(argv[1]);
	float threshold = atof(argv[2]);
	threshold = 0.0;
	float tp=0.0;
	float fp=0.0;
	float tn=0.0;
	int p=0;
	int num=0;
	float accuracy;
	
	vector<vector<string>> match_query;
	vector<float> accu;
	load_matchlist(match_list, match_query);
	ofstream infile;
	infile.open("fpr_tpr.txt",ios::trunc);
	for(int k=0;k<100;++k){
		threshold=1-float(k)/100;
		//cout<<threshold<<endl;
		tp=0.0;
		fp=0.0;
		tn=0.0;
		p=0.0;
		num=0.0;
	for(int i=0;i<match_query.size();++i){
		float cosine=atof(match_query[i][0].c_str());
		string match =match_query[i][1];
		if(match=="1"){
			p+=1;
		}
		if(cosine>threshold){
			if(match=="1"){
				tp+=1;
			}
			else{
				fp+=1;
			}
		}
		else{
			if(match=="0"){
				tn+=1;
			}
		}
		num+=1;
		
	}
		accuracy=(tp+tn)/num;
		accu.push_back(accuracy);
		infile<<fp/float(num-p)<<" "<<tp/float(p)<<" "<<accuracy<<endl;
        //cout<<"threshold = "<<threshold<<endl;	
		//cout<<"num = "<<num<<endl;
		//cout<<"p = "<<p<<endl;
		//cout<<"tp = "<<tp<<endl;
		//cout<<"fp = "<<fp<<endl;
		//cout<<"tn = "<<tn<<endl;
		//cout<<"TPR = "<<tp/float(p)<<endl;
		//cout<<"FPR = "<<fp/float(num-p)<<endl;
		//cout<<"accuracy = "<<accuracy<<endl;
		
	}
	infile.close();
	auto max_index=max_element(accu.begin(),accu.end());
	std::cout << "accuracy is " << *max_index<< " when thresh is " << 1-float(std::distance(accu.begin(), max_index))/100 << std::endl;
}