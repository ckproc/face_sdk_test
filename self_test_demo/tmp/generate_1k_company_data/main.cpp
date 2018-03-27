#include<iostream>
#include <vector>
#include <string>
#include "embed_face_feature.hpp"

using namespace std;

int main(int argc, const char *argv[]) {

	string db_dir = string(argv[1]);
	string model = string(argv[2]);
	string output_file = string(argv[3]);
  int isdb = embed_db_face(db_dir, model, output_file);
  if(isdb==-1){
    cout<<"embeding db faces failed."<<endl;
  }

  return 0;
}

