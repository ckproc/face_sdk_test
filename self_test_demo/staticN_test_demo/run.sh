cd ../staticN_generate_data 
./test ../1k_test_data/ft/query ./query.yml /workspace1/ckp/mclab_test/model /workspace1/ckp/mclab_test/model/sp.dat
#./test ../1k_test_data/company_data_clean/query ./query.yml /workspace1/ckp/mclab_test/model /workspace1/ckp/mclab_test/model/sp.dat
./test ../1k_test_data/ft/db ./db.yml /workspace1/ckp/mclab_test/model /workspace1/ckp/mclab_test/model/sp.dat
#./test ../1k_test_data/company_data_clean/db ./db.yml /workspace1/ckp/mclab_test/model /workspace1/ckp/mclab_test/model/sp.dat
cd ../staticN_test_demo
./test /workspace1/ckp/mclab_test/self_test_demo/staticN_generate_data/db.yml /workspace1/ckp/mclab_test/self_test_demo/staticN_generate_data/query.yml /workspace1/ckp/mclab_test/self_test_demo/1k_test_data/ft/matchlist.txt 0.6
#./test /workspace1/ckp/mclab_test/self_test_demo/staticN_generate_data/db.yml /workspace1/ckp/mclab_test/self_test_demo/staticN_generate_data/query.yml /workspace1/ckp/mclab_test/self_test_demo/1k_test_data/company_data_clean/matchlist.txt 0.6
vim fpr_tpr_recall.txt
#./test /workspace1/ckp/mclab_test/self_test_demo/generate_1k_company_data/cdb.yml /workspace1/ckp/mclab_test/self_test_demo/generate_1k_company_data/cquery.yml /workspace1/ckp/mclab_test/self_test_demo/1k_test_data/company_data_clean/matchlist.txt 0.65

