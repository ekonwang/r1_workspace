WORKSPACE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../..

cd ${WORKSPACE}
echo WORKSPACE=$(pwd)

mkdir -p ${WORKSPACE}/.temp/datasets/intergpt_geometry3k
cd ${WORKSPACE}/.temp/datasets/intergpt_geometry3k

wget https://lupantech.github.io/inter-gps/geometry3k/train.zip
wget https://lupantech.github.io/inter-gps/geometry3k/val.zip
wget https://lupantech.github.io/inter-gps/geometry3k/test.zip
wget https://lupantech.github.io/inter-gps/geometry3k/logic_forms.zip
wget https://lupantech.github.io/inter-gps/geometry3k/symbols.zip

unzip test.zip
unzip train.zip
unzip val.zip
unzip logic_forms.zip
unzip symbols.zip

