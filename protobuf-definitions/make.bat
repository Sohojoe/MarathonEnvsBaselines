# variables

# GRPC-TOOLS required. Install with `nuget install Grpc.Tools`. 
# Then un-comment and replace [DIRECTORY] with location of files.
# For example, on macOS, you might have something like:
# COMPILER=Grpc.Tools.1.14.1/tools/macosx_x64
# COMPILER=[DIRECTORY]

SRC_DIR=proto/mlagents/envs/communicator_objects
DST_DIR_C=../UnitySDK/Assets/ML-Agents/Scripts/CommunicatorObjects
DST_DIR_P=../ml-agents
PROTO_PATH=proto

PYTHON_PACKAGE=mlagents/envs/communicator_objects

# clean
rm -rf $DST_DIR_C
rm -rf $DST_DIR_P/$PYTHON_PACKAGE
mkdir -p $DST_DIR_C
mkdir -p $DST_DIR_P/$PYTHON_PACKAGE

# generate proto objects in python and C#

protoc --proto_path=proto --csharp_out=$DST_DIR_C $SRC_DIR/*.proto
protoc --proto_path=proto --python_out=$DST_DIR_P $SRC_DIR/*.proto 

# grpc 

GRPC=unity_to_external.proto

$COMPILER/protoc --proto_path=proto --csharp_out $DST_DIR_C --grpc_out $DST_DIR_C $SRC_DIR/$GRPC --plugin=protoc-gen-grpc=$COMPILER/grpc_csharp_plugin
python3 -m grpc_tools.protoc --proto_path=proto --python_out=$DST_DIR_P --grpc_python_out=$DST_DIR_P $SRC_DIR/$GRPC 


# Generate the init file for the python module
# rm -f $DST_DIR_P/$PYTHON_PACKAGE/__init__.py
for FILE in $DST_DIR_P/$PYTHON_PACKAGE/*.py
do 
FILE=${FILE##*/}
# echo from .$(basename $FILE) import \* >> $DST_DIR_P/$PYTHON_PACKAGE/__init__.py
echo from .${FILE%.py} import \* >> $DST_DIR_P/$PYTHON_PACKAGE/__init__.py
done

