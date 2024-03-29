��$
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

�
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758œ
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
�
Adam/v/conv2d_29/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/v/conv2d_29/bias
{
)Adam/v/conv2d_29/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_29/bias*
_output_shapes
:*
dtype0
�
Adam/m/conv2d_29/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/m/conv2d_29/bias
{
)Adam/m/conv2d_29/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_29/bias*
_output_shapes
:*
dtype0
�
Adam/v/conv2d_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/v/conv2d_29/kernel
�
+Adam/v/conv2d_29/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_29/kernel*&
_output_shapes
:@*
dtype0
�
Adam/m/conv2d_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/m/conv2d_29/kernel
�
+Adam/m/conv2d_29/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_29/kernel*&
_output_shapes
:@*
dtype0
�
Adam/v/conv2d_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/v/conv2d_28/bias
{
)Adam/v/conv2d_28/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_28/bias*
_output_shapes
:@*
dtype0
�
Adam/m/conv2d_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/m/conv2d_28/bias
{
)Adam/m/conv2d_28/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_28/bias*
_output_shapes
:@*
dtype0
�
Adam/v/conv2d_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/v/conv2d_28/kernel
�
+Adam/v/conv2d_28/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_28/kernel*&
_output_shapes
:@@*
dtype0
�
Adam/m/conv2d_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/m/conv2d_28/kernel
�
+Adam/m/conv2d_28/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_28/kernel*&
_output_shapes
:@@*
dtype0
�
Adam/v/conv2d_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/v/conv2d_27/bias
{
)Adam/v/conv2d_27/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_27/bias*
_output_shapes
:@*
dtype0
�
Adam/m/conv2d_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/m/conv2d_27/bias
{
)Adam/m/conv2d_27/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_27/bias*
_output_shapes
:@*
dtype0
�
Adam/v/conv2d_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�@*(
shared_nameAdam/v/conv2d_27/kernel
�
+Adam/v/conv2d_27/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_27/kernel*'
_output_shapes
:�@*
dtype0
�
Adam/m/conv2d_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�@*(
shared_nameAdam/m/conv2d_27/kernel
�
+Adam/m/conv2d_27/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_27/kernel*'
_output_shapes
:�@*
dtype0
�
Adam/v/conv2d_transpose_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name Adam/v/conv2d_transpose_5/bias
�
2Adam/v/conv2d_transpose_5/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_transpose_5/bias*
_output_shapes
:@*
dtype0
�
Adam/m/conv2d_transpose_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name Adam/m/conv2d_transpose_5/bias
�
2Adam/m/conv2d_transpose_5/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_transpose_5/bias*
_output_shapes
:@*
dtype0
�
 Adam/v/conv2d_transpose_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*1
shared_name" Adam/v/conv2d_transpose_5/kernel
�
4Adam/v/conv2d_transpose_5/kernel/Read/ReadVariableOpReadVariableOp Adam/v/conv2d_transpose_5/kernel*'
_output_shapes
:@�*
dtype0
�
 Adam/m/conv2d_transpose_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*1
shared_name" Adam/m/conv2d_transpose_5/kernel
�
4Adam/m/conv2d_transpose_5/kernel/Read/ReadVariableOpReadVariableOp Adam/m/conv2d_transpose_5/kernel*'
_output_shapes
:@�*
dtype0
�
Adam/v/conv2d_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/v/conv2d_26/bias
|
)Adam/v/conv2d_26/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_26/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/conv2d_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/m/conv2d_26/bias
|
)Adam/m/conv2d_26/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_26/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/conv2d_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/v/conv2d_26/kernel
�
+Adam/v/conv2d_26/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_26/kernel*(
_output_shapes
:��*
dtype0
�
Adam/m/conv2d_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/m/conv2d_26/kernel
�
+Adam/m/conv2d_26/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_26/kernel*(
_output_shapes
:��*
dtype0
�
Adam/v/conv2d_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/v/conv2d_25/bias
|
)Adam/v/conv2d_25/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_25/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/conv2d_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/m/conv2d_25/bias
|
)Adam/m/conv2d_25/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_25/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/conv2d_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/v/conv2d_25/kernel
�
+Adam/v/conv2d_25/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_25/kernel*(
_output_shapes
:��*
dtype0
�
Adam/m/conv2d_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/m/conv2d_25/kernel
�
+Adam/m/conv2d_25/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_25/kernel*(
_output_shapes
:��*
dtype0
�
Adam/v/conv2d_transpose_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*/
shared_name Adam/v/conv2d_transpose_4/bias
�
2Adam/v/conv2d_transpose_4/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_transpose_4/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/conv2d_transpose_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*/
shared_name Adam/m/conv2d_transpose_4/bias
�
2Adam/m/conv2d_transpose_4/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_transpose_4/bias*
_output_shapes	
:�*
dtype0
�
 Adam/v/conv2d_transpose_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*1
shared_name" Adam/v/conv2d_transpose_4/kernel
�
4Adam/v/conv2d_transpose_4/kernel/Read/ReadVariableOpReadVariableOp Adam/v/conv2d_transpose_4/kernel*(
_output_shapes
:��*
dtype0
�
 Adam/m/conv2d_transpose_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*1
shared_name" Adam/m/conv2d_transpose_4/kernel
�
4Adam/m/conv2d_transpose_4/kernel/Read/ReadVariableOpReadVariableOp Adam/m/conv2d_transpose_4/kernel*(
_output_shapes
:��*
dtype0
�
Adam/v/conv2d_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/v/conv2d_24/bias
|
)Adam/v/conv2d_24/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_24/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/conv2d_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/m/conv2d_24/bias
|
)Adam/m/conv2d_24/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_24/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/conv2d_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/v/conv2d_24/kernel
�
+Adam/v/conv2d_24/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_24/kernel*(
_output_shapes
:��*
dtype0
�
Adam/m/conv2d_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/m/conv2d_24/kernel
�
+Adam/m/conv2d_24/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_24/kernel*(
_output_shapes
:��*
dtype0
�
Adam/v/conv2d_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/v/conv2d_23/bias
|
)Adam/v/conv2d_23/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_23/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/conv2d_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/m/conv2d_23/bias
|
)Adam/m/conv2d_23/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_23/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/conv2d_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/v/conv2d_23/kernel
�
+Adam/v/conv2d_23/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_23/kernel*(
_output_shapes
:��*
dtype0
�
Adam/m/conv2d_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/m/conv2d_23/kernel
�
+Adam/m/conv2d_23/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_23/kernel*(
_output_shapes
:��*
dtype0
�
Adam/v/conv2d_transpose_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*/
shared_name Adam/v/conv2d_transpose_3/bias
�
2Adam/v/conv2d_transpose_3/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_transpose_3/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/conv2d_transpose_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*/
shared_name Adam/m/conv2d_transpose_3/bias
�
2Adam/m/conv2d_transpose_3/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_transpose_3/bias*
_output_shapes	
:�*
dtype0
�
 Adam/v/conv2d_transpose_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*1
shared_name" Adam/v/conv2d_transpose_3/kernel
�
4Adam/v/conv2d_transpose_3/kernel/Read/ReadVariableOpReadVariableOp Adam/v/conv2d_transpose_3/kernel*(
_output_shapes
:��*
dtype0
�
 Adam/m/conv2d_transpose_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*1
shared_name" Adam/m/conv2d_transpose_3/kernel
�
4Adam/m/conv2d_transpose_3/kernel/Read/ReadVariableOpReadVariableOp Adam/m/conv2d_transpose_3/kernel*(
_output_shapes
:��*
dtype0
�
Adam/v/conv2d_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/v/conv2d_22/bias
|
)Adam/v/conv2d_22/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_22/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/conv2d_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/m/conv2d_22/bias
|
)Adam/m/conv2d_22/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_22/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/conv2d_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/v/conv2d_22/kernel
�
+Adam/v/conv2d_22/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_22/kernel*(
_output_shapes
:��*
dtype0
�
Adam/m/conv2d_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/m/conv2d_22/kernel
�
+Adam/m/conv2d_22/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_22/kernel*(
_output_shapes
:��*
dtype0
�
Adam/v/conv2d_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/v/conv2d_21/bias
|
)Adam/v/conv2d_21/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_21/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/conv2d_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/m/conv2d_21/bias
|
)Adam/m/conv2d_21/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_21/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/conv2d_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/v/conv2d_21/kernel
�
+Adam/v/conv2d_21/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_21/kernel*(
_output_shapes
:��*
dtype0
�
Adam/m/conv2d_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/m/conv2d_21/kernel
�
+Adam/m/conv2d_21/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_21/kernel*(
_output_shapes
:��*
dtype0
�
Adam/v/conv2d_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/v/conv2d_20/bias
|
)Adam/v/conv2d_20/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_20/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/conv2d_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/m/conv2d_20/bias
|
)Adam/m/conv2d_20/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_20/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/conv2d_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/v/conv2d_20/kernel
�
+Adam/v/conv2d_20/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_20/kernel*(
_output_shapes
:��*
dtype0
�
Adam/m/conv2d_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/m/conv2d_20/kernel
�
+Adam/m/conv2d_20/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_20/kernel*(
_output_shapes
:��*
dtype0
�
Adam/v/conv2d_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/v/conv2d_19/bias
|
)Adam/v/conv2d_19/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_19/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/conv2d_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/m/conv2d_19/bias
|
)Adam/m/conv2d_19/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_19/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/conv2d_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/v/conv2d_19/kernel
�
+Adam/v/conv2d_19/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_19/kernel*(
_output_shapes
:��*
dtype0
�
Adam/m/conv2d_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/m/conv2d_19/kernel
�
+Adam/m/conv2d_19/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_19/kernel*(
_output_shapes
:��*
dtype0
�
Adam/v/conv2d_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/v/conv2d_18/bias
|
)Adam/v/conv2d_18/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_18/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/conv2d_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/m/conv2d_18/bias
|
)Adam/m/conv2d_18/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_18/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/conv2d_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/v/conv2d_18/kernel
�
+Adam/v/conv2d_18/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_18/kernel*(
_output_shapes
:��*
dtype0
�
Adam/m/conv2d_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/m/conv2d_18/kernel
�
+Adam/m/conv2d_18/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_18/kernel*(
_output_shapes
:��*
dtype0
�
Adam/v/conv2d_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/v/conv2d_17/bias
|
)Adam/v/conv2d_17/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_17/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/conv2d_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/m/conv2d_17/bias
|
)Adam/m/conv2d_17/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_17/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/conv2d_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*(
shared_nameAdam/v/conv2d_17/kernel
�
+Adam/v/conv2d_17/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_17/kernel*'
_output_shapes
:@�*
dtype0
�
Adam/m/conv2d_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*(
shared_nameAdam/m/conv2d_17/kernel
�
+Adam/m/conv2d_17/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_17/kernel*'
_output_shapes
:@�*
dtype0
�
Adam/v/conv2d_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/v/conv2d_16/bias
{
)Adam/v/conv2d_16/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_16/bias*
_output_shapes
:@*
dtype0
�
Adam/m/conv2d_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/m/conv2d_16/bias
{
)Adam/m/conv2d_16/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_16/bias*
_output_shapes
:@*
dtype0
�
Adam/v/conv2d_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/v/conv2d_16/kernel
�
+Adam/v/conv2d_16/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_16/kernel*&
_output_shapes
:@@*
dtype0
�
Adam/m/conv2d_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/m/conv2d_16/kernel
�
+Adam/m/conv2d_16/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_16/kernel*&
_output_shapes
:@@*
dtype0
�
Adam/v/conv2d_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/v/conv2d_15/bias
{
)Adam/v/conv2d_15/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_15/bias*
_output_shapes
:@*
dtype0
�
Adam/m/conv2d_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/m/conv2d_15/bias
{
)Adam/m/conv2d_15/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_15/bias*
_output_shapes
:@*
dtype0
�
Adam/v/conv2d_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/v/conv2d_15/kernel
�
+Adam/v/conv2d_15/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_15/kernel*&
_output_shapes
:@*
dtype0
�
Adam/m/conv2d_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/m/conv2d_15/kernel
�
+Adam/m/conv2d_15/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_15/kernel*&
_output_shapes
:@*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
t
conv2d_29/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_29/bias
m
"conv2d_29/bias/Read/ReadVariableOpReadVariableOpconv2d_29/bias*
_output_shapes
:*
dtype0
�
conv2d_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv2d_29/kernel
}
$conv2d_29/kernel/Read/ReadVariableOpReadVariableOpconv2d_29/kernel*&
_output_shapes
:@*
dtype0
t
conv2d_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_28/bias
m
"conv2d_28/bias/Read/ReadVariableOpReadVariableOpconv2d_28/bias*
_output_shapes
:@*
dtype0
�
conv2d_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_28/kernel
}
$conv2d_28/kernel/Read/ReadVariableOpReadVariableOpconv2d_28/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_27/bias
m
"conv2d_27/bias/Read/ReadVariableOpReadVariableOpconv2d_27/bias*
_output_shapes
:@*
dtype0
�
conv2d_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�@*!
shared_nameconv2d_27/kernel
~
$conv2d_27/kernel/Read/ReadVariableOpReadVariableOpconv2d_27/kernel*'
_output_shapes
:�@*
dtype0
�
conv2d_transpose_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameconv2d_transpose_5/bias

+conv2d_transpose_5/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_5/bias*
_output_shapes
:@*
dtype0
�
conv2d_transpose_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�**
shared_nameconv2d_transpose_5/kernel
�
-conv2d_transpose_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_5/kernel*'
_output_shapes
:@�*
dtype0
u
conv2d_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_26/bias
n
"conv2d_26/bias/Read/ReadVariableOpReadVariableOpconv2d_26/bias*
_output_shapes	
:�*
dtype0
�
conv2d_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*!
shared_nameconv2d_26/kernel

$conv2d_26/kernel/Read/ReadVariableOpReadVariableOpconv2d_26/kernel*(
_output_shapes
:��*
dtype0
u
conv2d_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_25/bias
n
"conv2d_25/bias/Read/ReadVariableOpReadVariableOpconv2d_25/bias*
_output_shapes	
:�*
dtype0
�
conv2d_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*!
shared_nameconv2d_25/kernel

$conv2d_25/kernel/Read/ReadVariableOpReadVariableOpconv2d_25/kernel*(
_output_shapes
:��*
dtype0
�
conv2d_transpose_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*(
shared_nameconv2d_transpose_4/bias
�
+conv2d_transpose_4/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_4/bias*
_output_shapes	
:�*
dtype0
�
conv2d_transpose_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��**
shared_nameconv2d_transpose_4/kernel
�
-conv2d_transpose_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_4/kernel*(
_output_shapes
:��*
dtype0
u
conv2d_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_24/bias
n
"conv2d_24/bias/Read/ReadVariableOpReadVariableOpconv2d_24/bias*
_output_shapes	
:�*
dtype0
�
conv2d_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*!
shared_nameconv2d_24/kernel

$conv2d_24/kernel/Read/ReadVariableOpReadVariableOpconv2d_24/kernel*(
_output_shapes
:��*
dtype0
u
conv2d_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_23/bias
n
"conv2d_23/bias/Read/ReadVariableOpReadVariableOpconv2d_23/bias*
_output_shapes	
:�*
dtype0
�
conv2d_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*!
shared_nameconv2d_23/kernel

$conv2d_23/kernel/Read/ReadVariableOpReadVariableOpconv2d_23/kernel*(
_output_shapes
:��*
dtype0
�
conv2d_transpose_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*(
shared_nameconv2d_transpose_3/bias
�
+conv2d_transpose_3/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_3/bias*
_output_shapes	
:�*
dtype0
�
conv2d_transpose_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��**
shared_nameconv2d_transpose_3/kernel
�
-conv2d_transpose_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_3/kernel*(
_output_shapes
:��*
dtype0
u
conv2d_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_22/bias
n
"conv2d_22/bias/Read/ReadVariableOpReadVariableOpconv2d_22/bias*
_output_shapes	
:�*
dtype0
�
conv2d_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*!
shared_nameconv2d_22/kernel

$conv2d_22/kernel/Read/ReadVariableOpReadVariableOpconv2d_22/kernel*(
_output_shapes
:��*
dtype0
u
conv2d_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_21/bias
n
"conv2d_21/bias/Read/ReadVariableOpReadVariableOpconv2d_21/bias*
_output_shapes	
:�*
dtype0
�
conv2d_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*!
shared_nameconv2d_21/kernel

$conv2d_21/kernel/Read/ReadVariableOpReadVariableOpconv2d_21/kernel*(
_output_shapes
:��*
dtype0
u
conv2d_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_20/bias
n
"conv2d_20/bias/Read/ReadVariableOpReadVariableOpconv2d_20/bias*
_output_shapes	
:�*
dtype0
�
conv2d_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*!
shared_nameconv2d_20/kernel

$conv2d_20/kernel/Read/ReadVariableOpReadVariableOpconv2d_20/kernel*(
_output_shapes
:��*
dtype0
u
conv2d_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_19/bias
n
"conv2d_19/bias/Read/ReadVariableOpReadVariableOpconv2d_19/bias*
_output_shapes	
:�*
dtype0
�
conv2d_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*!
shared_nameconv2d_19/kernel

$conv2d_19/kernel/Read/ReadVariableOpReadVariableOpconv2d_19/kernel*(
_output_shapes
:��*
dtype0
u
conv2d_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_18/bias
n
"conv2d_18/bias/Read/ReadVariableOpReadVariableOpconv2d_18/bias*
_output_shapes	
:�*
dtype0
�
conv2d_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*!
shared_nameconv2d_18/kernel

$conv2d_18/kernel/Read/ReadVariableOpReadVariableOpconv2d_18/kernel*(
_output_shapes
:��*
dtype0
u
conv2d_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_17/bias
n
"conv2d_17/bias/Read/ReadVariableOpReadVariableOpconv2d_17/bias*
_output_shapes	
:�*
dtype0
�
conv2d_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*!
shared_nameconv2d_17/kernel
~
$conv2d_17/kernel/Read/ReadVariableOpReadVariableOpconv2d_17/kernel*'
_output_shapes
:@�*
dtype0
t
conv2d_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_16/bias
m
"conv2d_16/bias/Read/ReadVariableOpReadVariableOpconv2d_16/bias*
_output_shapes
:@*
dtype0
�
conv2d_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_16/kernel
}
$conv2d_16/kernel/Read/ReadVariableOpReadVariableOpconv2d_16/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_15/bias
m
"conv2d_15/bias/Read/ReadVariableOpReadVariableOpconv2d_15/bias*
_output_shapes
:@*
dtype0
�
conv2d_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv2d_15/kernel
}
$conv2d_15/kernel/Read/ReadVariableOpReadVariableOpconv2d_15/kernel*&
_output_shapes
:@*
dtype0
�
serving_default_input_2Placeholder*/
_output_shapes
:���������@@*
dtype0*$
shape:���������@@
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2conv2d_15/kernelconv2d_15/biasconv2d_16/kernelconv2d_16/biasconv2d_17/kernelconv2d_17/biasconv2d_18/kernelconv2d_18/biasconv2d_19/kernelconv2d_19/biasconv2d_20/kernelconv2d_20/biasconv2d_21/kernelconv2d_21/biasconv2d_22/kernelconv2d_22/biasconv2d_transpose_3/kernelconv2d_transpose_3/biasconv2d_23/kernelconv2d_23/biasconv2d_24/kernelconv2d_24/biasconv2d_transpose_4/kernelconv2d_transpose_4/biasconv2d_25/kernelconv2d_25/biasconv2d_26/kernelconv2d_26/biasconv2d_transpose_5/kernelconv2d_transpose_5/biasconv2d_27/kernelconv2d_27/biasconv2d_28/kernelconv2d_28/biasconv2d_29/kernelconv2d_29/bias*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*F
_read_only_resource_inputs(
&$	
 !"#$*0
config_proto 

CPU

GPU2*0J 8� *-
f(R&
$__inference_signature_wrapper_308193

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
layer_with_weights-8
layer-12
layer-13
layer_with_weights-9
layer-14
layer_with_weights-10
layer-15
layer_with_weights-11
layer-16
layer-17
layer_with_weights-12
layer-18
layer_with_weights-13
layer-19
layer_with_weights-14
layer-20
layer-21
layer_with_weights-15
layer-22
layer_with_weights-16
layer-23
layer_with_weights-17
layer-24
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
 _default_save_signature
!	optimizer
"
signatures*
* 
�
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses

)kernel
*bias
 +_jit_compiled_convolution_op*
�
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses

2kernel
3bias
 4_jit_compiled_convolution_op*
�
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses* 
�
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses

Akernel
Bbias
 C_jit_compiled_convolution_op*
�
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses

Jkernel
Kbias
 L_jit_compiled_convolution_op*
�
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses* 
�
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses

Ykernel
Zbias
 [_jit_compiled_convolution_op*
�
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses

bkernel
cbias
 d_jit_compiled_convolution_op*
�
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses* 
�
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses

qkernel
rbias
 s_jit_compiled_convolution_op*
�
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses

zkernel
{bias
 |_jit_compiled_convolution_op*
�
}	variables
~trainable_variables
regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
)0
*1
22
33
A4
B5
J6
K7
Y8
Z9
b10
c11
q12
r13
z14
{15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35*
�
)0
*1
22
33
A4
B5
J6
K7
Y8
Z9
b10
c11
q12
r13
z14
{15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
 _default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
* 
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla*

�serving_default* 

)0
*1*

)0
*1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEconv2d_15/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_15/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

20
31*

20
31*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEconv2d_16/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_16/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

A0
B1*

A0
B1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEconv2d_17/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_17/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

J0
K1*

J0
K1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEconv2d_18/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_18/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

Y0
Z1*

Y0
Z1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEconv2d_19/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_19/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

b0
c1*

b0
c1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEconv2d_20/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_20/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

q0
r1*

q0
r1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEconv2d_21/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_21/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

z0
{1*

z0
{1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEconv2d_22/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_22/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
}	variables
~trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
ic
VARIABLE_VALUEconv2d_transpose_3/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEconv2d_transpose_3/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEconv2d_23/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_23/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
a[
VARIABLE_VALUEconv2d_24/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_24/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
jd
VARIABLE_VALUEconv2d_transpose_4/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEconv2d_transpose_4/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
a[
VARIABLE_VALUEconv2d_25/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_25/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
a[
VARIABLE_VALUEconv2d_26/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_26/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
jd
VARIABLE_VALUEconv2d_transpose_5/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEconv2d_transpose_5/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
a[
VARIABLE_VALUEconv2d_27/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_27/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
a[
VARIABLE_VALUEconv2d_28/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_28/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
a[
VARIABLE_VALUEconv2d_29/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_29/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24*

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56
�57
�58
�59
�60
�61
�62
�63
�64
�65
�66
�67
�68
�69
�70
�71
�72*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35*
�
�trace_0
�trace_1
�trace_2
�trace_3
�trace_4
�trace_5
�trace_6
�trace_7
�trace_8
�trace_9
�trace_10
�trace_11
�trace_12
�trace_13
�trace_14
�trace_15
�trace_16
�trace_17
�trace_18
�trace_19
�trace_20
�trace_21
�trace_22
�trace_23
�trace_24
�trace_25
�trace_26
�trace_27
�trace_28
�trace_29
�trace_30
�trace_31
�trace_32
�trace_33
�trace_34
�trace_35* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
b\
VARIABLE_VALUEAdam/m/conv2d_15/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_15/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_15/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_15/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_16/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_16/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_16/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_16/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_17/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv2d_17/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_17/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_17/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/conv2d_18/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv2d_18/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_18/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_18/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/conv2d_19/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv2d_19/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_19/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_19/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/conv2d_20/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv2d_20/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_20/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_20/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/conv2d_21/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv2d_21/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_21/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_21/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/conv2d_22/kernel2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv2d_22/kernel2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_22/bias2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_22/bias2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE Adam/m/conv2d_transpose_3/kernel2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE Adam/v/conv2d_transpose_3/kernel2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/m/conv2d_transpose_3/bias2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/v/conv2d_transpose_3/bias2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/conv2d_23/kernel2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv2d_23/kernel2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_23/bias2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_23/bias2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/conv2d_24/kernel2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv2d_24/kernel2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_24/bias2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_24/bias2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE Adam/m/conv2d_transpose_4/kernel2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE Adam/v/conv2d_transpose_4/kernel2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/m/conv2d_transpose_4/bias2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/v/conv2d_transpose_4/bias2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/conv2d_25/kernel2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv2d_25/kernel2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_25/bias2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_25/bias2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/conv2d_26/kernel2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv2d_26/kernel2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_26/bias2optimizer/_variables/55/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_26/bias2optimizer/_variables/56/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE Adam/m/conv2d_transpose_5/kernel2optimizer/_variables/57/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE Adam/v/conv2d_transpose_5/kernel2optimizer/_variables/58/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/m/conv2d_transpose_5/bias2optimizer/_variables/59/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/v/conv2d_transpose_5/bias2optimizer/_variables/60/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/conv2d_27/kernel2optimizer/_variables/61/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv2d_27/kernel2optimizer/_variables/62/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_27/bias2optimizer/_variables/63/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_27/bias2optimizer/_variables/64/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/conv2d_28/kernel2optimizer/_variables/65/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv2d_28/kernel2optimizer/_variables/66/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_28/bias2optimizer/_variables/67/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_28/bias2optimizer/_variables/68/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/conv2d_29/kernel2optimizer/_variables/69/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv2d_29/kernel2optimizer/_variables/70/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_29/bias2optimizer/_variables/71/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_29/bias2optimizer/_variables/72/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameconv2d_15/kernelconv2d_15/biasconv2d_16/kernelconv2d_16/biasconv2d_17/kernelconv2d_17/biasconv2d_18/kernelconv2d_18/biasconv2d_19/kernelconv2d_19/biasconv2d_20/kernelconv2d_20/biasconv2d_21/kernelconv2d_21/biasconv2d_22/kernelconv2d_22/biasconv2d_transpose_3/kernelconv2d_transpose_3/biasconv2d_23/kernelconv2d_23/biasconv2d_24/kernelconv2d_24/biasconv2d_transpose_4/kernelconv2d_transpose_4/biasconv2d_25/kernelconv2d_25/biasconv2d_26/kernelconv2d_26/biasconv2d_transpose_5/kernelconv2d_transpose_5/biasconv2d_27/kernelconv2d_27/biasconv2d_28/kernelconv2d_28/biasconv2d_29/kernelconv2d_29/bias	iterationlearning_rateAdam/m/conv2d_15/kernelAdam/v/conv2d_15/kernelAdam/m/conv2d_15/biasAdam/v/conv2d_15/biasAdam/m/conv2d_16/kernelAdam/v/conv2d_16/kernelAdam/m/conv2d_16/biasAdam/v/conv2d_16/biasAdam/m/conv2d_17/kernelAdam/v/conv2d_17/kernelAdam/m/conv2d_17/biasAdam/v/conv2d_17/biasAdam/m/conv2d_18/kernelAdam/v/conv2d_18/kernelAdam/m/conv2d_18/biasAdam/v/conv2d_18/biasAdam/m/conv2d_19/kernelAdam/v/conv2d_19/kernelAdam/m/conv2d_19/biasAdam/v/conv2d_19/biasAdam/m/conv2d_20/kernelAdam/v/conv2d_20/kernelAdam/m/conv2d_20/biasAdam/v/conv2d_20/biasAdam/m/conv2d_21/kernelAdam/v/conv2d_21/kernelAdam/m/conv2d_21/biasAdam/v/conv2d_21/biasAdam/m/conv2d_22/kernelAdam/v/conv2d_22/kernelAdam/m/conv2d_22/biasAdam/v/conv2d_22/bias Adam/m/conv2d_transpose_3/kernel Adam/v/conv2d_transpose_3/kernelAdam/m/conv2d_transpose_3/biasAdam/v/conv2d_transpose_3/biasAdam/m/conv2d_23/kernelAdam/v/conv2d_23/kernelAdam/m/conv2d_23/biasAdam/v/conv2d_23/biasAdam/m/conv2d_24/kernelAdam/v/conv2d_24/kernelAdam/m/conv2d_24/biasAdam/v/conv2d_24/bias Adam/m/conv2d_transpose_4/kernel Adam/v/conv2d_transpose_4/kernelAdam/m/conv2d_transpose_4/biasAdam/v/conv2d_transpose_4/biasAdam/m/conv2d_25/kernelAdam/v/conv2d_25/kernelAdam/m/conv2d_25/biasAdam/v/conv2d_25/biasAdam/m/conv2d_26/kernelAdam/v/conv2d_26/kernelAdam/m/conv2d_26/biasAdam/v/conv2d_26/bias Adam/m/conv2d_transpose_5/kernel Adam/v/conv2d_transpose_5/kernelAdam/m/conv2d_transpose_5/biasAdam/v/conv2d_transpose_5/biasAdam/m/conv2d_27/kernelAdam/v/conv2d_27/kernelAdam/m/conv2d_27/biasAdam/v/conv2d_27/biasAdam/m/conv2d_28/kernelAdam/v/conv2d_28/kernelAdam/m/conv2d_28/biasAdam/v/conv2d_28/biasAdam/m/conv2d_29/kernelAdam/v/conv2d_29/kernelAdam/m/conv2d_29/biasAdam/v/conv2d_29/biastotal_1count_1totalcountConst*
Tinx
v2t*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *(
f#R!
__inference__traced_save_309899
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_15/kernelconv2d_15/biasconv2d_16/kernelconv2d_16/biasconv2d_17/kernelconv2d_17/biasconv2d_18/kernelconv2d_18/biasconv2d_19/kernelconv2d_19/biasconv2d_20/kernelconv2d_20/biasconv2d_21/kernelconv2d_21/biasconv2d_22/kernelconv2d_22/biasconv2d_transpose_3/kernelconv2d_transpose_3/biasconv2d_23/kernelconv2d_23/biasconv2d_24/kernelconv2d_24/biasconv2d_transpose_4/kernelconv2d_transpose_4/biasconv2d_25/kernelconv2d_25/biasconv2d_26/kernelconv2d_26/biasconv2d_transpose_5/kernelconv2d_transpose_5/biasconv2d_27/kernelconv2d_27/biasconv2d_28/kernelconv2d_28/biasconv2d_29/kernelconv2d_29/bias	iterationlearning_rateAdam/m/conv2d_15/kernelAdam/v/conv2d_15/kernelAdam/m/conv2d_15/biasAdam/v/conv2d_15/biasAdam/m/conv2d_16/kernelAdam/v/conv2d_16/kernelAdam/m/conv2d_16/biasAdam/v/conv2d_16/biasAdam/m/conv2d_17/kernelAdam/v/conv2d_17/kernelAdam/m/conv2d_17/biasAdam/v/conv2d_17/biasAdam/m/conv2d_18/kernelAdam/v/conv2d_18/kernelAdam/m/conv2d_18/biasAdam/v/conv2d_18/biasAdam/m/conv2d_19/kernelAdam/v/conv2d_19/kernelAdam/m/conv2d_19/biasAdam/v/conv2d_19/biasAdam/m/conv2d_20/kernelAdam/v/conv2d_20/kernelAdam/m/conv2d_20/biasAdam/v/conv2d_20/biasAdam/m/conv2d_21/kernelAdam/v/conv2d_21/kernelAdam/m/conv2d_21/biasAdam/v/conv2d_21/biasAdam/m/conv2d_22/kernelAdam/v/conv2d_22/kernelAdam/m/conv2d_22/biasAdam/v/conv2d_22/bias Adam/m/conv2d_transpose_3/kernel Adam/v/conv2d_transpose_3/kernelAdam/m/conv2d_transpose_3/biasAdam/v/conv2d_transpose_3/biasAdam/m/conv2d_23/kernelAdam/v/conv2d_23/kernelAdam/m/conv2d_23/biasAdam/v/conv2d_23/biasAdam/m/conv2d_24/kernelAdam/v/conv2d_24/kernelAdam/m/conv2d_24/biasAdam/v/conv2d_24/bias Adam/m/conv2d_transpose_4/kernel Adam/v/conv2d_transpose_4/kernelAdam/m/conv2d_transpose_4/biasAdam/v/conv2d_transpose_4/biasAdam/m/conv2d_25/kernelAdam/v/conv2d_25/kernelAdam/m/conv2d_25/biasAdam/v/conv2d_25/biasAdam/m/conv2d_26/kernelAdam/v/conv2d_26/kernelAdam/m/conv2d_26/biasAdam/v/conv2d_26/bias Adam/m/conv2d_transpose_5/kernel Adam/v/conv2d_transpose_5/kernelAdam/m/conv2d_transpose_5/biasAdam/v/conv2d_transpose_5/biasAdam/m/conv2d_27/kernelAdam/v/conv2d_27/kernelAdam/m/conv2d_27/biasAdam/v/conv2d_27/biasAdam/m/conv2d_28/kernelAdam/v/conv2d_28/kernelAdam/m/conv2d_28/biasAdam/v/conv2d_28/biasAdam/m/conv2d_29/kernelAdam/v/conv2d_29/kernelAdam/m/conv2d_29/biasAdam/v/conv2d_29/biastotal_1count_1totalcount*~
Tinw
u2s*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__traced_restore_310251��
�
�
E__inference_conv2d_29_layer_call_and_return_conditional_losses_307325

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@^
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:���������@@b
IdentityIdentitySigmoid:y:0^NoOp*
T0*/
_output_shapes
:���������@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@@@
 
_user_specified_nameinputs
�
�
E__inference_conv2d_28_layer_call_and_return_conditional_losses_307308

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@@@
 
_user_specified_nameinputs
�
�
*__inference_conv2d_25_layer_call_fn_309046

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_25_layer_call_and_return_conditional_losses_307243x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������  �`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������  �: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������  �
 
_user_specified_nameinputs
�
�	
(__inference_model_1_layer_call_fn_307610
input_2!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@�
	unknown_4:	�%
	unknown_5:��
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�%
	unknown_9:��

unknown_10:	�&

unknown_11:��

unknown_12:	�&

unknown_13:��

unknown_14:	�&

unknown_15:��

unknown_16:	�&

unknown_17:��

unknown_18:	�&

unknown_19:��

unknown_20:	�&

unknown_21:��

unknown_22:	�&

unknown_23:��

unknown_24:	�&

unknown_25:��

unknown_26:	�%

unknown_27:@�

unknown_28:@%

unknown_29:�@

unknown_30:@$

unknown_31:@@

unknown_32:@$

unknown_33:@

unknown_34:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*F
_read_only_resource_inputs(
&$	
 !"#$*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_307535w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:���������@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������@@
!
_user_specified_name	input_2
�
L
#__inference__update_step_xla_297850
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:�: *
	_noinline(:E A

_output_shapes	
:�
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
�
3__inference_conv2d_transpose_3_layer_call_fn_308896

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_306929�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
��
�
C__inference_model_1_layer_call_and_return_conditional_losses_308522

inputsB
(conv2d_15_conv2d_readvariableop_resource:@7
)conv2d_15_biasadd_readvariableop_resource:@B
(conv2d_16_conv2d_readvariableop_resource:@@7
)conv2d_16_biasadd_readvariableop_resource:@C
(conv2d_17_conv2d_readvariableop_resource:@�8
)conv2d_17_biasadd_readvariableop_resource:	�D
(conv2d_18_conv2d_readvariableop_resource:��8
)conv2d_18_biasadd_readvariableop_resource:	�D
(conv2d_19_conv2d_readvariableop_resource:��8
)conv2d_19_biasadd_readvariableop_resource:	�D
(conv2d_20_conv2d_readvariableop_resource:��8
)conv2d_20_biasadd_readvariableop_resource:	�D
(conv2d_21_conv2d_readvariableop_resource:��8
)conv2d_21_biasadd_readvariableop_resource:	�D
(conv2d_22_conv2d_readvariableop_resource:��8
)conv2d_22_biasadd_readvariableop_resource:	�W
;conv2d_transpose_3_conv2d_transpose_readvariableop_resource:��A
2conv2d_transpose_3_biasadd_readvariableop_resource:	�D
(conv2d_23_conv2d_readvariableop_resource:��8
)conv2d_23_biasadd_readvariableop_resource:	�D
(conv2d_24_conv2d_readvariableop_resource:��8
)conv2d_24_biasadd_readvariableop_resource:	�W
;conv2d_transpose_4_conv2d_transpose_readvariableop_resource:��A
2conv2d_transpose_4_biasadd_readvariableop_resource:	�D
(conv2d_25_conv2d_readvariableop_resource:��8
)conv2d_25_biasadd_readvariableop_resource:	�D
(conv2d_26_conv2d_readvariableop_resource:��8
)conv2d_26_biasadd_readvariableop_resource:	�V
;conv2d_transpose_5_conv2d_transpose_readvariableop_resource:@�@
2conv2d_transpose_5_biasadd_readvariableop_resource:@C
(conv2d_27_conv2d_readvariableop_resource:�@7
)conv2d_27_biasadd_readvariableop_resource:@B
(conv2d_28_conv2d_readvariableop_resource:@@7
)conv2d_28_biasadd_readvariableop_resource:@B
(conv2d_29_conv2d_readvariableop_resource:@7
)conv2d_29_biasadd_readvariableop_resource:
identity�� conv2d_15/BiasAdd/ReadVariableOp�conv2d_15/Conv2D/ReadVariableOp� conv2d_16/BiasAdd/ReadVariableOp�conv2d_16/Conv2D/ReadVariableOp� conv2d_17/BiasAdd/ReadVariableOp�conv2d_17/Conv2D/ReadVariableOp� conv2d_18/BiasAdd/ReadVariableOp�conv2d_18/Conv2D/ReadVariableOp� conv2d_19/BiasAdd/ReadVariableOp�conv2d_19/Conv2D/ReadVariableOp� conv2d_20/BiasAdd/ReadVariableOp�conv2d_20/Conv2D/ReadVariableOp� conv2d_21/BiasAdd/ReadVariableOp�conv2d_21/Conv2D/ReadVariableOp� conv2d_22/BiasAdd/ReadVariableOp�conv2d_22/Conv2D/ReadVariableOp� conv2d_23/BiasAdd/ReadVariableOp�conv2d_23/Conv2D/ReadVariableOp� conv2d_24/BiasAdd/ReadVariableOp�conv2d_24/Conv2D/ReadVariableOp� conv2d_25/BiasAdd/ReadVariableOp�conv2d_25/Conv2D/ReadVariableOp� conv2d_26/BiasAdd/ReadVariableOp�conv2d_26/Conv2D/ReadVariableOp� conv2d_27/BiasAdd/ReadVariableOp�conv2d_27/Conv2D/ReadVariableOp� conv2d_28/BiasAdd/ReadVariableOp�conv2d_28/Conv2D/ReadVariableOp� conv2d_29/BiasAdd/ReadVariableOp�conv2d_29/Conv2D/ReadVariableOp�)conv2d_transpose_3/BiasAdd/ReadVariableOp�2conv2d_transpose_3/conv2d_transpose/ReadVariableOp�)conv2d_transpose_4/BiasAdd/ReadVariableOp�2conv2d_transpose_4/conv2d_transpose/ReadVariableOp�)conv2d_transpose_5/BiasAdd/ReadVariableOp�2conv2d_transpose_5/conv2d_transpose/ReadVariableOp�
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
conv2d_15/Conv2DConv2Dinputs'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@*
paddingSAME*
strides
�
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@l
conv2d_15/ReluReluconv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@@�
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
conv2d_16/Conv2DConv2Dconv2d_15/Relu:activations:0'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@*
paddingSAME*
strides
�
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@l
conv2d_16/ReluReluconv2d_16/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@@�
max_pooling2d_3/MaxPoolMaxPoolconv2d_16/Relu:activations:0*/
_output_shapes
:���������  @*
ksize
*
paddingVALID*
strides
�
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
conv2d_17/Conv2DConv2D max_pooling2d_3/MaxPool:output:0'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
�
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �m
conv2d_17/ReluReluconv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:���������  ��
conv2d_18/Conv2D/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_18/Conv2DConv2Dconv2d_17/Relu:activations:0'conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
�
 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0(conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �m
conv2d_18/ReluReluconv2d_18/BiasAdd:output:0*
T0*0
_output_shapes
:���������  ��
max_pooling2d_4/MaxPoolMaxPoolconv2d_18/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
�
conv2d_19/Conv2D/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_19/Conv2DConv2D max_pooling2d_4/MaxPool:output:0'conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������m
conv2d_19/ReluReluconv2d_19/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_20/Conv2DConv2Dconv2d_19/Relu:activations:0'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������m
conv2d_20/ReluReluconv2d_20/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
max_pooling2d_5/MaxPoolMaxPoolconv2d_20/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
�
conv2d_21/Conv2D/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_21/Conv2DConv2D max_pooling2d_5/MaxPool:output:0'conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
 conv2d_21/BiasAdd/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_21/BiasAddBiasAddconv2d_21/Conv2D:output:0(conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������m
conv2d_21/ReluReluconv2d_21/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_22/Conv2DConv2Dconv2d_21/Relu:activations:0'conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������m
conv2d_22/ReluReluconv2d_22/BiasAdd:output:0*
T0*0
_output_shapes
:����������r
conv2d_transpose_3/ShapeShapeconv2d_22/Relu:activations:0*
T0*
_output_shapes
::��p
&conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_3/strided_sliceStridedSlice!conv2d_transpose_3/Shape:output:0/conv2d_transpose_3/strided_slice/stack:output:01conv2d_transpose_3/strided_slice/stack_1:output:01conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value
B :��
conv2d_transpose_3/stackPack)conv2d_transpose_3/strided_slice:output:0#conv2d_transpose_3/stack/1:output:0#conv2d_transpose_3/stack/2:output:0#conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_3/strided_slice_1StridedSlice!conv2d_transpose_3/stack:output:01conv2d_transpose_3/strided_slice_1/stack:output:03conv2d_transpose_3/strided_slice_1/stack_1:output:03conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype0�
#conv2d_transpose_3/conv2d_transposeConv2DBackpropInput!conv2d_transpose_3/stack:output:0:conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0conv2d_22/Relu:activations:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
)conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_transpose_3/BiasAddBiasAdd,conv2d_transpose_3/conv2d_transpose:output:01conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������[
concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_3/concatConcatV2#conv2d_transpose_3/BiasAdd:output:0conv2d_20/Relu:activations:0"concatenate_3/concat/axis:output:0*
N*
T0*0
_output_shapes
:�����������
conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_23/Conv2DConv2Dconcatenate_3/concat:output:0'conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������m
conv2d_23/ReluReluconv2d_23/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_24/Conv2DConv2Dconv2d_23/Relu:activations:0'conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0(conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������m
conv2d_24/ReluReluconv2d_24/BiasAdd:output:0*
T0*0
_output_shapes
:����������r
conv2d_transpose_4/ShapeShapeconv2d_24/Relu:activations:0*
T0*
_output_shapes
::��p
&conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_4/strided_sliceStridedSlice!conv2d_transpose_4/Shape:output:0/conv2d_transpose_4/strided_slice/stack:output:01conv2d_transpose_4/strided_slice/stack_1:output:01conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B : \
conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B : ]
conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value
B :��
conv2d_transpose_4/stackPack)conv2d_transpose_4/strided_slice:output:0#conv2d_transpose_4/stack/1:output:0#conv2d_transpose_4/stack/2:output:0#conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_4/strided_slice_1StridedSlice!conv2d_transpose_4/stack:output:01conv2d_transpose_4/strided_slice_1/stack:output:03conv2d_transpose_4/strided_slice_1/stack_1:output:03conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_4_conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype0�
#conv2d_transpose_4/conv2d_transposeConv2DBackpropInput!conv2d_transpose_4/stack:output:0:conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0conv2d_24/Relu:activations:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
�
)conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_transpose_4/BiasAddBiasAdd,conv2d_transpose_4/conv2d_transpose:output:01conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �[
concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_4/concatConcatV2#conv2d_transpose_4/BiasAdd:output:0conv2d_18/Relu:activations:0"concatenate_4/concat/axis:output:0*
N*
T0*0
_output_shapes
:���������  ��
conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_25/Conv2DConv2Dconcatenate_4/concat:output:0'conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
�
 conv2d_25/BiasAdd/ReadVariableOpReadVariableOp)conv2d_25_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_25/BiasAddBiasAddconv2d_25/Conv2D:output:0(conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �m
conv2d_25/ReluReluconv2d_25/BiasAdd:output:0*
T0*0
_output_shapes
:���������  ��
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_26/Conv2DConv2Dconv2d_25/Relu:activations:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
�
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �m
conv2d_26/ReluReluconv2d_26/BiasAdd:output:0*
T0*0
_output_shapes
:���������  �r
conv2d_transpose_5/ShapeShapeconv2d_26/Relu:activations:0*
T0*
_output_shapes
::��p
&conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_5/strided_sliceStridedSlice!conv2d_transpose_5/Shape:output:0/conv2d_transpose_5/strided_slice/stack:output:01conv2d_transpose_5/strided_slice/stack_1:output:01conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@\
conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@\
conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@�
conv2d_transpose_5/stackPack)conv2d_transpose_5/strided_slice:output:0#conv2d_transpose_5/stack/1:output:0#conv2d_transpose_5/stack/2:output:0#conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_5/strided_slice_1StridedSlice!conv2d_transpose_5/stack:output:01conv2d_transpose_5/strided_slice_1/stack:output:03conv2d_transpose_5/strided_slice_1/stack_1:output:03conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_5_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
#conv2d_transpose_5/conv2d_transposeConv2DBackpropInput!conv2d_transpose_5/stack:output:0:conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0conv2d_26/Relu:activations:0*
T0*/
_output_shapes
:���������@@@*
paddingSAME*
strides
�
)conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_transpose_5/BiasAddBiasAdd,conv2d_transpose_5/conv2d_transpose:output:01conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@[
concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_5/concatConcatV2#conv2d_transpose_5/BiasAdd:output:0conv2d_16/Relu:activations:0"concatenate_5/concat/axis:output:0*
N*
T0*0
_output_shapes
:���������@@��
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*'
_output_shapes
:�@*
dtype0�
conv2d_27/Conv2DConv2Dconcatenate_5/concat:output:0'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@*
paddingSAME*
strides
�
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@l
conv2d_27/ReluReluconv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@@�
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
conv2d_28/Conv2DConv2Dconv2d_27/Relu:activations:0'conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@*
paddingSAME*
strides
�
 conv2d_28/BiasAdd/ReadVariableOpReadVariableOp)conv2d_28_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_28/BiasAddBiasAddconv2d_28/Conv2D:output:0(conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@l
conv2d_28/ReluReluconv2d_28/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@@�
conv2d_29/Conv2D/ReadVariableOpReadVariableOp(conv2d_29_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
conv2d_29/Conv2DConv2Dconv2d_28/Relu:activations:0'conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingVALID*
strides
�
 conv2d_29/BiasAdd/ReadVariableOpReadVariableOp)conv2d_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_29/BiasAddBiasAddconv2d_29/Conv2D:output:0(conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@r
conv2d_29/SigmoidSigmoidconv2d_29/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@l
IdentityIdentityconv2d_29/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:���������@@�

NoOpNoOp!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp!^conv2d_18/BiasAdd/ReadVariableOp ^conv2d_18/Conv2D/ReadVariableOp!^conv2d_19/BiasAdd/ReadVariableOp ^conv2d_19/Conv2D/ReadVariableOp!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp!^conv2d_21/BiasAdd/ReadVariableOp ^conv2d_21/Conv2D/ReadVariableOp!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp!^conv2d_23/BiasAdd/ReadVariableOp ^conv2d_23/Conv2D/ReadVariableOp!^conv2d_24/BiasAdd/ReadVariableOp ^conv2d_24/Conv2D/ReadVariableOp!^conv2d_25/BiasAdd/ReadVariableOp ^conv2d_25/Conv2D/ReadVariableOp!^conv2d_26/BiasAdd/ReadVariableOp ^conv2d_26/Conv2D/ReadVariableOp!^conv2d_27/BiasAdd/ReadVariableOp ^conv2d_27/Conv2D/ReadVariableOp!^conv2d_28/BiasAdd/ReadVariableOp ^conv2d_28/Conv2D/ReadVariableOp!^conv2d_29/BiasAdd/ReadVariableOp ^conv2d_29/Conv2D/ReadVariableOp*^conv2d_transpose_3/BiasAdd/ReadVariableOp3^conv2d_transpose_3/conv2d_transpose/ReadVariableOp*^conv2d_transpose_4/BiasAdd/ReadVariableOp3^conv2d_transpose_4/conv2d_transpose/ReadVariableOp*^conv2d_transpose_5/BiasAdd/ReadVariableOp3^conv2d_transpose_5/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:���������@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp2D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp2D
 conv2d_18/BiasAdd/ReadVariableOp conv2d_18/BiasAdd/ReadVariableOp2B
conv2d_18/Conv2D/ReadVariableOpconv2d_18/Conv2D/ReadVariableOp2D
 conv2d_19/BiasAdd/ReadVariableOp conv2d_19/BiasAdd/ReadVariableOp2B
conv2d_19/Conv2D/ReadVariableOpconv2d_19/Conv2D/ReadVariableOp2D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp2D
 conv2d_21/BiasAdd/ReadVariableOp conv2d_21/BiasAdd/ReadVariableOp2B
conv2d_21/Conv2D/ReadVariableOpconv2d_21/Conv2D/ReadVariableOp2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp2D
 conv2d_23/BiasAdd/ReadVariableOp conv2d_23/BiasAdd/ReadVariableOp2B
conv2d_23/Conv2D/ReadVariableOpconv2d_23/Conv2D/ReadVariableOp2D
 conv2d_24/BiasAdd/ReadVariableOp conv2d_24/BiasAdd/ReadVariableOp2B
conv2d_24/Conv2D/ReadVariableOpconv2d_24/Conv2D/ReadVariableOp2D
 conv2d_25/BiasAdd/ReadVariableOp conv2d_25/BiasAdd/ReadVariableOp2B
conv2d_25/Conv2D/ReadVariableOpconv2d_25/Conv2D/ReadVariableOp2D
 conv2d_26/BiasAdd/ReadVariableOp conv2d_26/BiasAdd/ReadVariableOp2B
conv2d_26/Conv2D/ReadVariableOpconv2d_26/Conv2D/ReadVariableOp2D
 conv2d_27/BiasAdd/ReadVariableOp conv2d_27/BiasAdd/ReadVariableOp2B
conv2d_27/Conv2D/ReadVariableOpconv2d_27/Conv2D/ReadVariableOp2D
 conv2d_28/BiasAdd/ReadVariableOp conv2d_28/BiasAdd/ReadVariableOp2B
conv2d_28/Conv2D/ReadVariableOpconv2d_28/Conv2D/ReadVariableOp2D
 conv2d_29/BiasAdd/ReadVariableOp conv2d_29/BiasAdd/ReadVariableOp2B
conv2d_29/Conv2D/ReadVariableOpconv2d_29/Conv2D/ReadVariableOp2V
)conv2d_transpose_3/BiasAdd/ReadVariableOp)conv2d_transpose_3/BiasAdd/ReadVariableOp2h
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_4/BiasAdd/ReadVariableOp)conv2d_transpose_4/BiasAdd/ReadVariableOp2h
2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_5/BiasAdd/ReadVariableOp)conv2d_transpose_5/BiasAdd/ReadVariableOp2h
2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2conv2d_transpose_5/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
�
E__inference_conv2d_22_layer_call_and_return_conditional_losses_308887

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
s
I__inference_concatenate_5_layer_call_and_return_conditional_losses_307278

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :~
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:���������@@�`
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:���������@@�"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������@@@:���������@@@:W S
/
_output_shapes
:���������@@@
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������@@@
 
_user_specified_nameinputs
��
�!
!__inference__wrapped_model_306859
input_2J
0model_1_conv2d_15_conv2d_readvariableop_resource:@?
1model_1_conv2d_15_biasadd_readvariableop_resource:@J
0model_1_conv2d_16_conv2d_readvariableop_resource:@@?
1model_1_conv2d_16_biasadd_readvariableop_resource:@K
0model_1_conv2d_17_conv2d_readvariableop_resource:@�@
1model_1_conv2d_17_biasadd_readvariableop_resource:	�L
0model_1_conv2d_18_conv2d_readvariableop_resource:��@
1model_1_conv2d_18_biasadd_readvariableop_resource:	�L
0model_1_conv2d_19_conv2d_readvariableop_resource:��@
1model_1_conv2d_19_biasadd_readvariableop_resource:	�L
0model_1_conv2d_20_conv2d_readvariableop_resource:��@
1model_1_conv2d_20_biasadd_readvariableop_resource:	�L
0model_1_conv2d_21_conv2d_readvariableop_resource:��@
1model_1_conv2d_21_biasadd_readvariableop_resource:	�L
0model_1_conv2d_22_conv2d_readvariableop_resource:��@
1model_1_conv2d_22_biasadd_readvariableop_resource:	�_
Cmodel_1_conv2d_transpose_3_conv2d_transpose_readvariableop_resource:��I
:model_1_conv2d_transpose_3_biasadd_readvariableop_resource:	�L
0model_1_conv2d_23_conv2d_readvariableop_resource:��@
1model_1_conv2d_23_biasadd_readvariableop_resource:	�L
0model_1_conv2d_24_conv2d_readvariableop_resource:��@
1model_1_conv2d_24_biasadd_readvariableop_resource:	�_
Cmodel_1_conv2d_transpose_4_conv2d_transpose_readvariableop_resource:��I
:model_1_conv2d_transpose_4_biasadd_readvariableop_resource:	�L
0model_1_conv2d_25_conv2d_readvariableop_resource:��@
1model_1_conv2d_25_biasadd_readvariableop_resource:	�L
0model_1_conv2d_26_conv2d_readvariableop_resource:��@
1model_1_conv2d_26_biasadd_readvariableop_resource:	�^
Cmodel_1_conv2d_transpose_5_conv2d_transpose_readvariableop_resource:@�H
:model_1_conv2d_transpose_5_biasadd_readvariableop_resource:@K
0model_1_conv2d_27_conv2d_readvariableop_resource:�@?
1model_1_conv2d_27_biasadd_readvariableop_resource:@J
0model_1_conv2d_28_conv2d_readvariableop_resource:@@?
1model_1_conv2d_28_biasadd_readvariableop_resource:@J
0model_1_conv2d_29_conv2d_readvariableop_resource:@?
1model_1_conv2d_29_biasadd_readvariableop_resource:
identity��(model_1/conv2d_15/BiasAdd/ReadVariableOp�'model_1/conv2d_15/Conv2D/ReadVariableOp�(model_1/conv2d_16/BiasAdd/ReadVariableOp�'model_1/conv2d_16/Conv2D/ReadVariableOp�(model_1/conv2d_17/BiasAdd/ReadVariableOp�'model_1/conv2d_17/Conv2D/ReadVariableOp�(model_1/conv2d_18/BiasAdd/ReadVariableOp�'model_1/conv2d_18/Conv2D/ReadVariableOp�(model_1/conv2d_19/BiasAdd/ReadVariableOp�'model_1/conv2d_19/Conv2D/ReadVariableOp�(model_1/conv2d_20/BiasAdd/ReadVariableOp�'model_1/conv2d_20/Conv2D/ReadVariableOp�(model_1/conv2d_21/BiasAdd/ReadVariableOp�'model_1/conv2d_21/Conv2D/ReadVariableOp�(model_1/conv2d_22/BiasAdd/ReadVariableOp�'model_1/conv2d_22/Conv2D/ReadVariableOp�(model_1/conv2d_23/BiasAdd/ReadVariableOp�'model_1/conv2d_23/Conv2D/ReadVariableOp�(model_1/conv2d_24/BiasAdd/ReadVariableOp�'model_1/conv2d_24/Conv2D/ReadVariableOp�(model_1/conv2d_25/BiasAdd/ReadVariableOp�'model_1/conv2d_25/Conv2D/ReadVariableOp�(model_1/conv2d_26/BiasAdd/ReadVariableOp�'model_1/conv2d_26/Conv2D/ReadVariableOp�(model_1/conv2d_27/BiasAdd/ReadVariableOp�'model_1/conv2d_27/Conv2D/ReadVariableOp�(model_1/conv2d_28/BiasAdd/ReadVariableOp�'model_1/conv2d_28/Conv2D/ReadVariableOp�(model_1/conv2d_29/BiasAdd/ReadVariableOp�'model_1/conv2d_29/Conv2D/ReadVariableOp�1model_1/conv2d_transpose_3/BiasAdd/ReadVariableOp�:model_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOp�1model_1/conv2d_transpose_4/BiasAdd/ReadVariableOp�:model_1/conv2d_transpose_4/conv2d_transpose/ReadVariableOp�1model_1/conv2d_transpose_5/BiasAdd/ReadVariableOp�:model_1/conv2d_transpose_5/conv2d_transpose/ReadVariableOp�
'model_1/conv2d_15/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
model_1/conv2d_15/Conv2DConv2Dinput_2/model_1/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@*
paddingSAME*
strides
�
(model_1/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model_1/conv2d_15/BiasAddBiasAdd!model_1/conv2d_15/Conv2D:output:00model_1/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@|
model_1/conv2d_15/ReluRelu"model_1/conv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@@�
'model_1/conv2d_16/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
model_1/conv2d_16/Conv2DConv2D$model_1/conv2d_15/Relu:activations:0/model_1/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@*
paddingSAME*
strides
�
(model_1/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model_1/conv2d_16/BiasAddBiasAdd!model_1/conv2d_16/Conv2D:output:00model_1/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@|
model_1/conv2d_16/ReluRelu"model_1/conv2d_16/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@@�
model_1/max_pooling2d_3/MaxPoolMaxPool$model_1/conv2d_16/Relu:activations:0*/
_output_shapes
:���������  @*
ksize
*
paddingVALID*
strides
�
'model_1/conv2d_17/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_17_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
model_1/conv2d_17/Conv2DConv2D(model_1/max_pooling2d_3/MaxPool:output:0/model_1/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
�
(model_1/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model_1/conv2d_17/BiasAddBiasAdd!model_1/conv2d_17/Conv2D:output:00model_1/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �}
model_1/conv2d_17/ReluRelu"model_1/conv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:���������  ��
'model_1/conv2d_18/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_18_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
model_1/conv2d_18/Conv2DConv2D$model_1/conv2d_17/Relu:activations:0/model_1/conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
�
(model_1/conv2d_18/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_18_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model_1/conv2d_18/BiasAddBiasAdd!model_1/conv2d_18/Conv2D:output:00model_1/conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �}
model_1/conv2d_18/ReluRelu"model_1/conv2d_18/BiasAdd:output:0*
T0*0
_output_shapes
:���������  ��
model_1/max_pooling2d_4/MaxPoolMaxPool$model_1/conv2d_18/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
�
'model_1/conv2d_19/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_19_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
model_1/conv2d_19/Conv2DConv2D(model_1/max_pooling2d_4/MaxPool:output:0/model_1/conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
(model_1/conv2d_19/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model_1/conv2d_19/BiasAddBiasAdd!model_1/conv2d_19/Conv2D:output:00model_1/conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������}
model_1/conv2d_19/ReluRelu"model_1/conv2d_19/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
'model_1/conv2d_20/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_20_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
model_1/conv2d_20/Conv2DConv2D$model_1/conv2d_19/Relu:activations:0/model_1/conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
(model_1/conv2d_20/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_20_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model_1/conv2d_20/BiasAddBiasAdd!model_1/conv2d_20/Conv2D:output:00model_1/conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������}
model_1/conv2d_20/ReluRelu"model_1/conv2d_20/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
model_1/max_pooling2d_5/MaxPoolMaxPool$model_1/conv2d_20/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
�
'model_1/conv2d_21/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_21_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
model_1/conv2d_21/Conv2DConv2D(model_1/max_pooling2d_5/MaxPool:output:0/model_1/conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
(model_1/conv2d_21/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_21_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model_1/conv2d_21/BiasAddBiasAdd!model_1/conv2d_21/Conv2D:output:00model_1/conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������}
model_1/conv2d_21/ReluRelu"model_1/conv2d_21/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
'model_1/conv2d_22/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_22_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
model_1/conv2d_22/Conv2DConv2D$model_1/conv2d_21/Relu:activations:0/model_1/conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
(model_1/conv2d_22/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_22_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model_1/conv2d_22/BiasAddBiasAdd!model_1/conv2d_22/Conv2D:output:00model_1/conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������}
model_1/conv2d_22/ReluRelu"model_1/conv2d_22/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
 model_1/conv2d_transpose_3/ShapeShape$model_1/conv2d_22/Relu:activations:0*
T0*
_output_shapes
::��x
.model_1/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0model_1/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0model_1/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(model_1/conv2d_transpose_3/strided_sliceStridedSlice)model_1/conv2d_transpose_3/Shape:output:07model_1/conv2d_transpose_3/strided_slice/stack:output:09model_1/conv2d_transpose_3/strided_slice/stack_1:output:09model_1/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"model_1/conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :d
"model_1/conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :e
"model_1/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value
B :��
 model_1/conv2d_transpose_3/stackPack1model_1/conv2d_transpose_3/strided_slice:output:0+model_1/conv2d_transpose_3/stack/1:output:0+model_1/conv2d_transpose_3/stack/2:output:0+model_1/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:z
0model_1/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2model_1/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2model_1/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
*model_1/conv2d_transpose_3/strided_slice_1StridedSlice)model_1/conv2d_transpose_3/stack:output:09model_1/conv2d_transpose_3/strided_slice_1/stack:output:0;model_1/conv2d_transpose_3/strided_slice_1/stack_1:output:0;model_1/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
:model_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpCmodel_1_conv2d_transpose_3_conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype0�
+model_1/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput)model_1/conv2d_transpose_3/stack:output:0Bmodel_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0$model_1/conv2d_22/Relu:activations:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
1model_1/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp:model_1_conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
"model_1/conv2d_transpose_3/BiasAddBiasAdd4model_1/conv2d_transpose_3/conv2d_transpose:output:09model_1/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������c
!model_1/concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model_1/concatenate_3/concatConcatV2+model_1/conv2d_transpose_3/BiasAdd:output:0$model_1/conv2d_20/Relu:activations:0*model_1/concatenate_3/concat/axis:output:0*
N*
T0*0
_output_shapes
:�����������
'model_1/conv2d_23/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_23_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
model_1/conv2d_23/Conv2DConv2D%model_1/concatenate_3/concat:output:0/model_1/conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
(model_1/conv2d_23/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_23_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model_1/conv2d_23/BiasAddBiasAdd!model_1/conv2d_23/Conv2D:output:00model_1/conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������}
model_1/conv2d_23/ReluRelu"model_1/conv2d_23/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
'model_1/conv2d_24/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_24_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
model_1/conv2d_24/Conv2DConv2D$model_1/conv2d_23/Relu:activations:0/model_1/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
(model_1/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_24_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model_1/conv2d_24/BiasAddBiasAdd!model_1/conv2d_24/Conv2D:output:00model_1/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������}
model_1/conv2d_24/ReluRelu"model_1/conv2d_24/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
 model_1/conv2d_transpose_4/ShapeShape$model_1/conv2d_24/Relu:activations:0*
T0*
_output_shapes
::��x
.model_1/conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0model_1/conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0model_1/conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(model_1/conv2d_transpose_4/strided_sliceStridedSlice)model_1/conv2d_transpose_4/Shape:output:07model_1/conv2d_transpose_4/strided_slice/stack:output:09model_1/conv2d_transpose_4/strided_slice/stack_1:output:09model_1/conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"model_1/conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B : d
"model_1/conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B : e
"model_1/conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value
B :��
 model_1/conv2d_transpose_4/stackPack1model_1/conv2d_transpose_4/strided_slice:output:0+model_1/conv2d_transpose_4/stack/1:output:0+model_1/conv2d_transpose_4/stack/2:output:0+model_1/conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:z
0model_1/conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2model_1/conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2model_1/conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
*model_1/conv2d_transpose_4/strided_slice_1StridedSlice)model_1/conv2d_transpose_4/stack:output:09model_1/conv2d_transpose_4/strided_slice_1/stack:output:0;model_1/conv2d_transpose_4/strided_slice_1/stack_1:output:0;model_1/conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
:model_1/conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOpCmodel_1_conv2d_transpose_4_conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype0�
+model_1/conv2d_transpose_4/conv2d_transposeConv2DBackpropInput)model_1/conv2d_transpose_4/stack:output:0Bmodel_1/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0$model_1/conv2d_24/Relu:activations:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
�
1model_1/conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp:model_1_conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
"model_1/conv2d_transpose_4/BiasAddBiasAdd4model_1/conv2d_transpose_4/conv2d_transpose:output:09model_1/conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �c
!model_1/concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model_1/concatenate_4/concatConcatV2+model_1/conv2d_transpose_4/BiasAdd:output:0$model_1/conv2d_18/Relu:activations:0*model_1/concatenate_4/concat/axis:output:0*
N*
T0*0
_output_shapes
:���������  ��
'model_1/conv2d_25/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_25_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
model_1/conv2d_25/Conv2DConv2D%model_1/concatenate_4/concat:output:0/model_1/conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
�
(model_1/conv2d_25/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_25_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model_1/conv2d_25/BiasAddBiasAdd!model_1/conv2d_25/Conv2D:output:00model_1/conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �}
model_1/conv2d_25/ReluRelu"model_1/conv2d_25/BiasAdd:output:0*
T0*0
_output_shapes
:���������  ��
'model_1/conv2d_26/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_26_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
model_1/conv2d_26/Conv2DConv2D$model_1/conv2d_25/Relu:activations:0/model_1/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
�
(model_1/conv2d_26/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_26_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model_1/conv2d_26/BiasAddBiasAdd!model_1/conv2d_26/Conv2D:output:00model_1/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �}
model_1/conv2d_26/ReluRelu"model_1/conv2d_26/BiasAdd:output:0*
T0*0
_output_shapes
:���������  ��
 model_1/conv2d_transpose_5/ShapeShape$model_1/conv2d_26/Relu:activations:0*
T0*
_output_shapes
::��x
.model_1/conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0model_1/conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0model_1/conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(model_1/conv2d_transpose_5/strided_sliceStridedSlice)model_1/conv2d_transpose_5/Shape:output:07model_1/conv2d_transpose_5/strided_slice/stack:output:09model_1/conv2d_transpose_5/strided_slice/stack_1:output:09model_1/conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"model_1/conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@d
"model_1/conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@d
"model_1/conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@�
 model_1/conv2d_transpose_5/stackPack1model_1/conv2d_transpose_5/strided_slice:output:0+model_1/conv2d_transpose_5/stack/1:output:0+model_1/conv2d_transpose_5/stack/2:output:0+model_1/conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:z
0model_1/conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2model_1/conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2model_1/conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
*model_1/conv2d_transpose_5/strided_slice_1StridedSlice)model_1/conv2d_transpose_5/stack:output:09model_1/conv2d_transpose_5/strided_slice_1/stack:output:0;model_1/conv2d_transpose_5/strided_slice_1/stack_1:output:0;model_1/conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
:model_1/conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOpCmodel_1_conv2d_transpose_5_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
+model_1/conv2d_transpose_5/conv2d_transposeConv2DBackpropInput)model_1/conv2d_transpose_5/stack:output:0Bmodel_1/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0$model_1/conv2d_26/Relu:activations:0*
T0*/
_output_shapes
:���������@@@*
paddingSAME*
strides
�
1model_1/conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp:model_1_conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
"model_1/conv2d_transpose_5/BiasAddBiasAdd4model_1/conv2d_transpose_5/conv2d_transpose:output:09model_1/conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@c
!model_1/concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model_1/concatenate_5/concatConcatV2+model_1/conv2d_transpose_5/BiasAdd:output:0$model_1/conv2d_16/Relu:activations:0*model_1/concatenate_5/concat/axis:output:0*
N*
T0*0
_output_shapes
:���������@@��
'model_1/conv2d_27/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_27_conv2d_readvariableop_resource*'
_output_shapes
:�@*
dtype0�
model_1/conv2d_27/Conv2DConv2D%model_1/concatenate_5/concat:output:0/model_1/conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@*
paddingSAME*
strides
�
(model_1/conv2d_27/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model_1/conv2d_27/BiasAddBiasAdd!model_1/conv2d_27/Conv2D:output:00model_1/conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@|
model_1/conv2d_27/ReluRelu"model_1/conv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@@�
'model_1/conv2d_28/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
model_1/conv2d_28/Conv2DConv2D$model_1/conv2d_27/Relu:activations:0/model_1/conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@*
paddingSAME*
strides
�
(model_1/conv2d_28/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_28_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model_1/conv2d_28/BiasAddBiasAdd!model_1/conv2d_28/Conv2D:output:00model_1/conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@|
model_1/conv2d_28/ReluRelu"model_1/conv2d_28/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@@�
'model_1/conv2d_29/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_29_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
model_1/conv2d_29/Conv2DConv2D$model_1/conv2d_28/Relu:activations:0/model_1/conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingVALID*
strides
�
(model_1/conv2d_29/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_1/conv2d_29/BiasAddBiasAdd!model_1/conv2d_29/Conv2D:output:00model_1/conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@�
model_1/conv2d_29/SigmoidSigmoid"model_1/conv2d_29/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@t
IdentityIdentitymodel_1/conv2d_29/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:���������@@�
NoOpNoOp)^model_1/conv2d_15/BiasAdd/ReadVariableOp(^model_1/conv2d_15/Conv2D/ReadVariableOp)^model_1/conv2d_16/BiasAdd/ReadVariableOp(^model_1/conv2d_16/Conv2D/ReadVariableOp)^model_1/conv2d_17/BiasAdd/ReadVariableOp(^model_1/conv2d_17/Conv2D/ReadVariableOp)^model_1/conv2d_18/BiasAdd/ReadVariableOp(^model_1/conv2d_18/Conv2D/ReadVariableOp)^model_1/conv2d_19/BiasAdd/ReadVariableOp(^model_1/conv2d_19/Conv2D/ReadVariableOp)^model_1/conv2d_20/BiasAdd/ReadVariableOp(^model_1/conv2d_20/Conv2D/ReadVariableOp)^model_1/conv2d_21/BiasAdd/ReadVariableOp(^model_1/conv2d_21/Conv2D/ReadVariableOp)^model_1/conv2d_22/BiasAdd/ReadVariableOp(^model_1/conv2d_22/Conv2D/ReadVariableOp)^model_1/conv2d_23/BiasAdd/ReadVariableOp(^model_1/conv2d_23/Conv2D/ReadVariableOp)^model_1/conv2d_24/BiasAdd/ReadVariableOp(^model_1/conv2d_24/Conv2D/ReadVariableOp)^model_1/conv2d_25/BiasAdd/ReadVariableOp(^model_1/conv2d_25/Conv2D/ReadVariableOp)^model_1/conv2d_26/BiasAdd/ReadVariableOp(^model_1/conv2d_26/Conv2D/ReadVariableOp)^model_1/conv2d_27/BiasAdd/ReadVariableOp(^model_1/conv2d_27/Conv2D/ReadVariableOp)^model_1/conv2d_28/BiasAdd/ReadVariableOp(^model_1/conv2d_28/Conv2D/ReadVariableOp)^model_1/conv2d_29/BiasAdd/ReadVariableOp(^model_1/conv2d_29/Conv2D/ReadVariableOp2^model_1/conv2d_transpose_3/BiasAdd/ReadVariableOp;^model_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2^model_1/conv2d_transpose_4/BiasAdd/ReadVariableOp;^model_1/conv2d_transpose_4/conv2d_transpose/ReadVariableOp2^model_1/conv2d_transpose_5/BiasAdd/ReadVariableOp;^model_1/conv2d_transpose_5/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:���������@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2T
(model_1/conv2d_15/BiasAdd/ReadVariableOp(model_1/conv2d_15/BiasAdd/ReadVariableOp2R
'model_1/conv2d_15/Conv2D/ReadVariableOp'model_1/conv2d_15/Conv2D/ReadVariableOp2T
(model_1/conv2d_16/BiasAdd/ReadVariableOp(model_1/conv2d_16/BiasAdd/ReadVariableOp2R
'model_1/conv2d_16/Conv2D/ReadVariableOp'model_1/conv2d_16/Conv2D/ReadVariableOp2T
(model_1/conv2d_17/BiasAdd/ReadVariableOp(model_1/conv2d_17/BiasAdd/ReadVariableOp2R
'model_1/conv2d_17/Conv2D/ReadVariableOp'model_1/conv2d_17/Conv2D/ReadVariableOp2T
(model_1/conv2d_18/BiasAdd/ReadVariableOp(model_1/conv2d_18/BiasAdd/ReadVariableOp2R
'model_1/conv2d_18/Conv2D/ReadVariableOp'model_1/conv2d_18/Conv2D/ReadVariableOp2T
(model_1/conv2d_19/BiasAdd/ReadVariableOp(model_1/conv2d_19/BiasAdd/ReadVariableOp2R
'model_1/conv2d_19/Conv2D/ReadVariableOp'model_1/conv2d_19/Conv2D/ReadVariableOp2T
(model_1/conv2d_20/BiasAdd/ReadVariableOp(model_1/conv2d_20/BiasAdd/ReadVariableOp2R
'model_1/conv2d_20/Conv2D/ReadVariableOp'model_1/conv2d_20/Conv2D/ReadVariableOp2T
(model_1/conv2d_21/BiasAdd/ReadVariableOp(model_1/conv2d_21/BiasAdd/ReadVariableOp2R
'model_1/conv2d_21/Conv2D/ReadVariableOp'model_1/conv2d_21/Conv2D/ReadVariableOp2T
(model_1/conv2d_22/BiasAdd/ReadVariableOp(model_1/conv2d_22/BiasAdd/ReadVariableOp2R
'model_1/conv2d_22/Conv2D/ReadVariableOp'model_1/conv2d_22/Conv2D/ReadVariableOp2T
(model_1/conv2d_23/BiasAdd/ReadVariableOp(model_1/conv2d_23/BiasAdd/ReadVariableOp2R
'model_1/conv2d_23/Conv2D/ReadVariableOp'model_1/conv2d_23/Conv2D/ReadVariableOp2T
(model_1/conv2d_24/BiasAdd/ReadVariableOp(model_1/conv2d_24/BiasAdd/ReadVariableOp2R
'model_1/conv2d_24/Conv2D/ReadVariableOp'model_1/conv2d_24/Conv2D/ReadVariableOp2T
(model_1/conv2d_25/BiasAdd/ReadVariableOp(model_1/conv2d_25/BiasAdd/ReadVariableOp2R
'model_1/conv2d_25/Conv2D/ReadVariableOp'model_1/conv2d_25/Conv2D/ReadVariableOp2T
(model_1/conv2d_26/BiasAdd/ReadVariableOp(model_1/conv2d_26/BiasAdd/ReadVariableOp2R
'model_1/conv2d_26/Conv2D/ReadVariableOp'model_1/conv2d_26/Conv2D/ReadVariableOp2T
(model_1/conv2d_27/BiasAdd/ReadVariableOp(model_1/conv2d_27/BiasAdd/ReadVariableOp2R
'model_1/conv2d_27/Conv2D/ReadVariableOp'model_1/conv2d_27/Conv2D/ReadVariableOp2T
(model_1/conv2d_28/BiasAdd/ReadVariableOp(model_1/conv2d_28/BiasAdd/ReadVariableOp2R
'model_1/conv2d_28/Conv2D/ReadVariableOp'model_1/conv2d_28/Conv2D/ReadVariableOp2T
(model_1/conv2d_29/BiasAdd/ReadVariableOp(model_1/conv2d_29/BiasAdd/ReadVariableOp2R
'model_1/conv2d_29/Conv2D/ReadVariableOp'model_1/conv2d_29/Conv2D/ReadVariableOp2f
1model_1/conv2d_transpose_3/BiasAdd/ReadVariableOp1model_1/conv2d_transpose_3/BiasAdd/ReadVariableOp2x
:model_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:model_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2f
1model_1/conv2d_transpose_4/BiasAdd/ReadVariableOp1model_1/conv2d_transpose_4/BiasAdd/ReadVariableOp2x
:model_1/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:model_1/conv2d_transpose_4/conv2d_transpose/ReadVariableOp2f
1model_1/conv2d_transpose_5/BiasAdd/ReadVariableOp1model_1/conv2d_transpose_5/BiasAdd/ReadVariableOp2x
:model_1/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:model_1/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:X T
/
_output_shapes
:���������@@
!
_user_specified_name	input_2
�
�
*__inference_conv2d_21_layer_call_fn_308856

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_21_layer_call_and_return_conditional_losses_307147x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
E__inference_conv2d_18_layer_call_and_return_conditional_losses_307094

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������  �j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������  �w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������  �: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������  �
 
_user_specified_nameinputs
� 
�
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_307017

inputsC
(conv2d_transpose_readvariableop_resource:@�-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
E__inference_conv2d_15_layer_call_and_return_conditional_losses_307042

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
�
E__inference_conv2d_16_layer_call_and_return_conditional_losses_307059

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@@@
 
_user_specified_nameinputs
�
�
*__inference_conv2d_23_layer_call_fn_308951

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_23_layer_call_and_return_conditional_losses_307195x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
L
#__inference__update_step_xla_297870
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:�: *
	_noinline(:E A

_output_shapes	
:�
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
� 
�
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_306973

inputsD
(conv2d_transpose_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :�y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������z
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�s
�
C__inference_model_1_layer_call_and_return_conditional_losses_307332
input_2*
conv2d_15_307043:@
conv2d_15_307045:@*
conv2d_16_307060:@@
conv2d_16_307062:@+
conv2d_17_307078:@�
conv2d_17_307080:	�,
conv2d_18_307095:��
conv2d_18_307097:	�,
conv2d_19_307113:��
conv2d_19_307115:	�,
conv2d_20_307130:��
conv2d_20_307132:	�,
conv2d_21_307148:��
conv2d_21_307150:	�,
conv2d_22_307165:��
conv2d_22_307167:	�5
conv2d_transpose_3_307170:��(
conv2d_transpose_3_307172:	�,
conv2d_23_307196:��
conv2d_23_307198:	�,
conv2d_24_307213:��
conv2d_24_307215:	�5
conv2d_transpose_4_307218:��(
conv2d_transpose_4_307220:	�,
conv2d_25_307244:��
conv2d_25_307246:	�,
conv2d_26_307261:��
conv2d_26_307263:	�4
conv2d_transpose_5_307266:@�'
conv2d_transpose_5_307268:@+
conv2d_27_307292:�@
conv2d_27_307294:@*
conv2d_28_307309:@@
conv2d_28_307311:@*
conv2d_29_307326:@
conv2d_29_307328:
identity��!conv2d_15/StatefulPartitionedCall�!conv2d_16/StatefulPartitionedCall�!conv2d_17/StatefulPartitionedCall�!conv2d_18/StatefulPartitionedCall�!conv2d_19/StatefulPartitionedCall�!conv2d_20/StatefulPartitionedCall�!conv2d_21/StatefulPartitionedCall�!conv2d_22/StatefulPartitionedCall�!conv2d_23/StatefulPartitionedCall�!conv2d_24/StatefulPartitionedCall�!conv2d_25/StatefulPartitionedCall�!conv2d_26/StatefulPartitionedCall�!conv2d_27/StatefulPartitionedCall�!conv2d_28/StatefulPartitionedCall�!conv2d_29/StatefulPartitionedCall�*conv2d_transpose_3/StatefulPartitionedCall�*conv2d_transpose_4/StatefulPartitionedCall�*conv2d_transpose_5/StatefulPartitionedCall�
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_15_307043conv2d_15_307045*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_307042�
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0conv2d_16_307060conv2d_16_307062*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_16_layer_call_and_return_conditional_losses_307059�
max_pooling2d_3/PartitionedCallPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_306865�
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv2d_17_307078conv2d_17_307080*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_17_layer_call_and_return_conditional_losses_307077�
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0conv2d_18_307095conv2d_18_307097*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_18_layer_call_and_return_conditional_losses_307094�
max_pooling2d_4/PartitionedCallPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_306877�
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_19_307113conv2d_19_307115*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_19_layer_call_and_return_conditional_losses_307112�
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0conv2d_20_307130conv2d_20_307132*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_20_layer_call_and_return_conditional_losses_307129�
max_pooling2d_5/PartitionedCallPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_306889�
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0conv2d_21_307148conv2d_21_307150*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_21_layer_call_and_return_conditional_losses_307147�
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0conv2d_22_307165conv2d_22_307167*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_22_layer_call_and_return_conditional_losses_307164�
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0conv2d_transpose_3_307170conv2d_transpose_3_307172*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_306929�
concatenate_3/PartitionedCallPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0*conv2d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_concatenate_3_layer_call_and_return_conditional_losses_307182�
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0conv2d_23_307196conv2d_23_307198*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_23_layer_call_and_return_conditional_losses_307195�
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0conv2d_24_307213conv2d_24_307215*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_24_layer_call_and_return_conditional_losses_307212�
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0conv2d_transpose_4_307218conv2d_transpose_4_307220*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_306973�
concatenate_4/PartitionedCallPartitionedCall3conv2d_transpose_4/StatefulPartitionedCall:output:0*conv2d_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_concatenate_4_layer_call_and_return_conditional_losses_307230�
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0conv2d_25_307244conv2d_25_307246*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_25_layer_call_and_return_conditional_losses_307243�
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0conv2d_26_307261conv2d_26_307263*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_26_layer_call_and_return_conditional_losses_307260�
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0conv2d_transpose_5_307266conv2d_transpose_5_307268*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_307017�
concatenate_5/PartitionedCallPartitionedCall3conv2d_transpose_5/StatefulPartitionedCall:output:0*conv2d_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_concatenate_5_layer_call_and_return_conditional_losses_307278�
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0conv2d_27_307292conv2d_27_307294*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_27_layer_call_and_return_conditional_losses_307291�
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0conv2d_28_307309conv2d_28_307311*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_28_layer_call_and_return_conditional_losses_307308�
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0conv2d_29_307326conv2d_29_307328*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_29_layer_call_and_return_conditional_losses_307325�
IdentityIdentity*conv2d_29/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@�
NoOpNoOp"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall+^conv2d_transpose_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:���������@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall:X T
/
_output_shapes
:���������@@
!
_user_specified_name	input_2
� 
�
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_309119

inputsC
(conv2d_transpose_readvariableop_resource:@�-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
W
#__inference__update_step_xla_297825
gradient"
variable:@@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:@@: *
	_noinline(:P L
&
_output_shapes
:@@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
u
I__inference_concatenate_5_layer_call_and_return_conditional_losses_309132
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:���������@@�`
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:���������@@�"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������@@@:���������@@@:Y U
/
_output_shapes
:���������@@@
"
_user_specified_name
inputs_0:YU
/
_output_shapes
:���������@@@
"
_user_specified_name
inputs_1
�
W
#__inference__update_step_xla_297985
gradient"
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:@: *
	_noinline(:P L
&
_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
�
*__inference_conv2d_19_layer_call_fn_308806

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_19_layer_call_and_return_conditional_losses_307112x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
L
#__inference__update_step_xla_297900
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:�: *
	_noinline(:E A

_output_shapes	
:�
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
L
0__inference_max_pooling2d_3_layer_call_fn_308742

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_306865�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�s
�
C__inference_model_1_layer_call_and_return_conditional_losses_307432
input_2*
conv2d_15_307335:@
conv2d_15_307337:@*
conv2d_16_307340:@@
conv2d_16_307342:@+
conv2d_17_307346:@�
conv2d_17_307348:	�,
conv2d_18_307351:��
conv2d_18_307353:	�,
conv2d_19_307357:��
conv2d_19_307359:	�,
conv2d_20_307362:��
conv2d_20_307364:	�,
conv2d_21_307368:��
conv2d_21_307370:	�,
conv2d_22_307373:��
conv2d_22_307375:	�5
conv2d_transpose_3_307378:��(
conv2d_transpose_3_307380:	�,
conv2d_23_307384:��
conv2d_23_307386:	�,
conv2d_24_307389:��
conv2d_24_307391:	�5
conv2d_transpose_4_307394:��(
conv2d_transpose_4_307396:	�,
conv2d_25_307400:��
conv2d_25_307402:	�,
conv2d_26_307405:��
conv2d_26_307407:	�4
conv2d_transpose_5_307410:@�'
conv2d_transpose_5_307412:@+
conv2d_27_307416:�@
conv2d_27_307418:@*
conv2d_28_307421:@@
conv2d_28_307423:@*
conv2d_29_307426:@
conv2d_29_307428:
identity��!conv2d_15/StatefulPartitionedCall�!conv2d_16/StatefulPartitionedCall�!conv2d_17/StatefulPartitionedCall�!conv2d_18/StatefulPartitionedCall�!conv2d_19/StatefulPartitionedCall�!conv2d_20/StatefulPartitionedCall�!conv2d_21/StatefulPartitionedCall�!conv2d_22/StatefulPartitionedCall�!conv2d_23/StatefulPartitionedCall�!conv2d_24/StatefulPartitionedCall�!conv2d_25/StatefulPartitionedCall�!conv2d_26/StatefulPartitionedCall�!conv2d_27/StatefulPartitionedCall�!conv2d_28/StatefulPartitionedCall�!conv2d_29/StatefulPartitionedCall�*conv2d_transpose_3/StatefulPartitionedCall�*conv2d_transpose_4/StatefulPartitionedCall�*conv2d_transpose_5/StatefulPartitionedCall�
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_15_307335conv2d_15_307337*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_307042�
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0conv2d_16_307340conv2d_16_307342*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_16_layer_call_and_return_conditional_losses_307059�
max_pooling2d_3/PartitionedCallPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_306865�
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv2d_17_307346conv2d_17_307348*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_17_layer_call_and_return_conditional_losses_307077�
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0conv2d_18_307351conv2d_18_307353*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_18_layer_call_and_return_conditional_losses_307094�
max_pooling2d_4/PartitionedCallPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_306877�
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_19_307357conv2d_19_307359*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_19_layer_call_and_return_conditional_losses_307112�
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0conv2d_20_307362conv2d_20_307364*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_20_layer_call_and_return_conditional_losses_307129�
max_pooling2d_5/PartitionedCallPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_306889�
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0conv2d_21_307368conv2d_21_307370*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_21_layer_call_and_return_conditional_losses_307147�
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0conv2d_22_307373conv2d_22_307375*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_22_layer_call_and_return_conditional_losses_307164�
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0conv2d_transpose_3_307378conv2d_transpose_3_307380*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_306929�
concatenate_3/PartitionedCallPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0*conv2d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_concatenate_3_layer_call_and_return_conditional_losses_307182�
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0conv2d_23_307384conv2d_23_307386*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_23_layer_call_and_return_conditional_losses_307195�
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0conv2d_24_307389conv2d_24_307391*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_24_layer_call_and_return_conditional_losses_307212�
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0conv2d_transpose_4_307394conv2d_transpose_4_307396*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_306973�
concatenate_4/PartitionedCallPartitionedCall3conv2d_transpose_4/StatefulPartitionedCall:output:0*conv2d_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_concatenate_4_layer_call_and_return_conditional_losses_307230�
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0conv2d_25_307400conv2d_25_307402*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_25_layer_call_and_return_conditional_losses_307243�
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0conv2d_26_307405conv2d_26_307407*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_26_layer_call_and_return_conditional_losses_307260�
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0conv2d_transpose_5_307410conv2d_transpose_5_307412*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_307017�
concatenate_5/PartitionedCallPartitionedCall3conv2d_transpose_5/StatefulPartitionedCall:output:0*conv2d_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_concatenate_5_layer_call_and_return_conditional_losses_307278�
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0conv2d_27_307416conv2d_27_307418*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_27_layer_call_and_return_conditional_losses_307291�
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0conv2d_28_307421conv2d_28_307423*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_28_layer_call_and_return_conditional_losses_307308�
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0conv2d_29_307426conv2d_29_307428*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_29_layer_call_and_return_conditional_losses_307325�
IdentityIdentity*conv2d_29/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@�
NoOpNoOp"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall+^conv2d_transpose_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:���������@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall:X T
/
_output_shapes
:���������@@
!
_user_specified_name	input_2
�
�
E__inference_conv2d_20_layer_call_and_return_conditional_losses_308837

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
E__inference_conv2d_20_layer_call_and_return_conditional_losses_307129

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�s
�
C__inference_model_1_layer_call_and_return_conditional_losses_307712

inputs*
conv2d_15_307615:@
conv2d_15_307617:@*
conv2d_16_307620:@@
conv2d_16_307622:@+
conv2d_17_307626:@�
conv2d_17_307628:	�,
conv2d_18_307631:��
conv2d_18_307633:	�,
conv2d_19_307637:��
conv2d_19_307639:	�,
conv2d_20_307642:��
conv2d_20_307644:	�,
conv2d_21_307648:��
conv2d_21_307650:	�,
conv2d_22_307653:��
conv2d_22_307655:	�5
conv2d_transpose_3_307658:��(
conv2d_transpose_3_307660:	�,
conv2d_23_307664:��
conv2d_23_307666:	�,
conv2d_24_307669:��
conv2d_24_307671:	�5
conv2d_transpose_4_307674:��(
conv2d_transpose_4_307676:	�,
conv2d_25_307680:��
conv2d_25_307682:	�,
conv2d_26_307685:��
conv2d_26_307687:	�4
conv2d_transpose_5_307690:@�'
conv2d_transpose_5_307692:@+
conv2d_27_307696:�@
conv2d_27_307698:@*
conv2d_28_307701:@@
conv2d_28_307703:@*
conv2d_29_307706:@
conv2d_29_307708:
identity��!conv2d_15/StatefulPartitionedCall�!conv2d_16/StatefulPartitionedCall�!conv2d_17/StatefulPartitionedCall�!conv2d_18/StatefulPartitionedCall�!conv2d_19/StatefulPartitionedCall�!conv2d_20/StatefulPartitionedCall�!conv2d_21/StatefulPartitionedCall�!conv2d_22/StatefulPartitionedCall�!conv2d_23/StatefulPartitionedCall�!conv2d_24/StatefulPartitionedCall�!conv2d_25/StatefulPartitionedCall�!conv2d_26/StatefulPartitionedCall�!conv2d_27/StatefulPartitionedCall�!conv2d_28/StatefulPartitionedCall�!conv2d_29/StatefulPartitionedCall�*conv2d_transpose_3/StatefulPartitionedCall�*conv2d_transpose_4/StatefulPartitionedCall�*conv2d_transpose_5/StatefulPartitionedCall�
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_15_307615conv2d_15_307617*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_307042�
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0conv2d_16_307620conv2d_16_307622*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_16_layer_call_and_return_conditional_losses_307059�
max_pooling2d_3/PartitionedCallPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_306865�
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv2d_17_307626conv2d_17_307628*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_17_layer_call_and_return_conditional_losses_307077�
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0conv2d_18_307631conv2d_18_307633*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_18_layer_call_and_return_conditional_losses_307094�
max_pooling2d_4/PartitionedCallPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_306877�
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_19_307637conv2d_19_307639*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_19_layer_call_and_return_conditional_losses_307112�
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0conv2d_20_307642conv2d_20_307644*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_20_layer_call_and_return_conditional_losses_307129�
max_pooling2d_5/PartitionedCallPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_306889�
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0conv2d_21_307648conv2d_21_307650*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_21_layer_call_and_return_conditional_losses_307147�
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0conv2d_22_307653conv2d_22_307655*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_22_layer_call_and_return_conditional_losses_307164�
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0conv2d_transpose_3_307658conv2d_transpose_3_307660*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_306929�
concatenate_3/PartitionedCallPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0*conv2d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_concatenate_3_layer_call_and_return_conditional_losses_307182�
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0conv2d_23_307664conv2d_23_307666*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_23_layer_call_and_return_conditional_losses_307195�
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0conv2d_24_307669conv2d_24_307671*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_24_layer_call_and_return_conditional_losses_307212�
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0conv2d_transpose_4_307674conv2d_transpose_4_307676*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_306973�
concatenate_4/PartitionedCallPartitionedCall3conv2d_transpose_4/StatefulPartitionedCall:output:0*conv2d_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_concatenate_4_layer_call_and_return_conditional_losses_307230�
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0conv2d_25_307680conv2d_25_307682*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_25_layer_call_and_return_conditional_losses_307243�
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0conv2d_26_307685conv2d_26_307687*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_26_layer_call_and_return_conditional_losses_307260�
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0conv2d_transpose_5_307690conv2d_transpose_5_307692*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_307017�
concatenate_5/PartitionedCallPartitionedCall3conv2d_transpose_5/StatefulPartitionedCall:output:0*conv2d_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_concatenate_5_layer_call_and_return_conditional_losses_307278�
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0conv2d_27_307696conv2d_27_307698*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_27_layer_call_and_return_conditional_losses_307291�
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0conv2d_28_307701conv2d_28_307703*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_28_layer_call_and_return_conditional_losses_307308�
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0conv2d_29_307706conv2d_29_307708*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_29_layer_call_and_return_conditional_losses_307325�
IdentityIdentity*conv2d_29/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@�
NoOpNoOp"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall+^conv2d_transpose_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:���������@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
L
#__inference__update_step_xla_297840
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:�: *
	_noinline(:E A

_output_shapes	
:�
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
Z
.__inference_concatenate_4_layer_call_fn_309030
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_concatenate_4_layer_call_and_return_conditional_losses_307230i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:���������  �"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������  �:���������  �:Z V
0
_output_shapes
:���������  �
"
_user_specified_name
inputs_0:ZV
0
_output_shapes
:���������  �
"
_user_specified_name
inputs_1
�
�
*__inference_conv2d_22_layer_call_fn_308876

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_22_layer_call_and_return_conditional_losses_307164x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
L
#__inference__update_step_xla_297910
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:�: *
	_noinline(:E A

_output_shapes	
:�
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
�
*__inference_conv2d_26_layer_call_fn_309066

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_26_layer_call_and_return_conditional_losses_307260x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������  �`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������  �: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������  �
 
_user_specified_nameinputs
�
�
E__inference_conv2d_21_layer_call_and_return_conditional_losses_307147

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
K
#__inference__update_step_xla_297820
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
L
#__inference__update_step_xla_297920
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:�: *
	_noinline(:E A

_output_shapes	
:�
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
X
#__inference__update_step_xla_297955
gradient#
variable:@�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*(
_input_shapes
:@�: *
	_noinline(:Q M
'
_output_shapes
:@�
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
Y
#__inference__update_step_xla_297885
gradient$
variable:��*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*)
_input_shapes
:��: *
	_noinline(:R N
(
_output_shapes
:��
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
Y
#__inference__update_step_xla_297915
gradient$
variable:��*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*)
_input_shapes
:��: *
	_noinline(:R N
(
_output_shapes
:��
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
�	
(__inference_model_1_layer_call_fn_307787
input_2!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@�
	unknown_4:	�%
	unknown_5:��
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�%
	unknown_9:��

unknown_10:	�&

unknown_11:��

unknown_12:	�&

unknown_13:��

unknown_14:	�&

unknown_15:��

unknown_16:	�&

unknown_17:��

unknown_18:	�&

unknown_19:��

unknown_20:	�&

unknown_21:��

unknown_22:	�&

unknown_23:��

unknown_24:	�&

unknown_25:��

unknown_26:	�%

unknown_27:@�

unknown_28:@%

unknown_29:�@

unknown_30:@$

unknown_31:@@

unknown_32:@$

unknown_33:@

unknown_34:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*F
_read_only_resource_inputs(
&$	
 !"#$*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_307712w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:���������@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������@@
!
_user_specified_name	input_2
�
Y
#__inference__update_step_xla_297935
gradient$
variable:��*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*)
_input_shapes
:��: *
	_noinline(:R N
(
_output_shapes
:��
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
u
I__inference_concatenate_4_layer_call_and_return_conditional_losses_309037
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:���������  �`
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:���������  �"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������  �:���������  �:Z V
0
_output_shapes
:���������  �
"
_user_specified_name
inputs_0:ZV
0
_output_shapes
:���������  �
"
_user_specified_name
inputs_1
�
K
#__inference__update_step_xla_297990
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
�
E__inference_conv2d_28_layer_call_and_return_conditional_losses_309172

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@@@
 
_user_specified_nameinputs
�
�
E__inference_conv2d_27_layer_call_and_return_conditional_losses_307291

inputs9
conv2d_readvariableop_resource:�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:�@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
*__inference_conv2d_29_layer_call_fn_309181

inputs!
unknown:@
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_29_layer_call_and_return_conditional_losses_307325w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@@@
 
_user_specified_nameinputs
�
�
E__inference_conv2d_29_layer_call_and_return_conditional_losses_309192

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@^
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:���������@@b
IdentityIdentitySigmoid:y:0^NoOp*
T0*/
_output_shapes
:���������@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@@@
 
_user_specified_nameinputs
�
g
K__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_306889

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
*__inference_conv2d_15_layer_call_fn_308706

inputs!
unknown:@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_307042w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
Y
#__inference__update_step_xla_297875
gradient$
variable:��*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*)
_input_shapes
:��: *
	_noinline(:R N
(
_output_shapes
:��
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
�
E__inference_conv2d_19_layer_call_and_return_conditional_losses_308817

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
E__inference_conv2d_25_layer_call_and_return_conditional_losses_307243

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������  �j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������  �w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������  �: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������  �
 
_user_specified_nameinputs
�
g
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_308747

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
��
�
C__inference_model_1_layer_call_and_return_conditional_losses_308697

inputsB
(conv2d_15_conv2d_readvariableop_resource:@7
)conv2d_15_biasadd_readvariableop_resource:@B
(conv2d_16_conv2d_readvariableop_resource:@@7
)conv2d_16_biasadd_readvariableop_resource:@C
(conv2d_17_conv2d_readvariableop_resource:@�8
)conv2d_17_biasadd_readvariableop_resource:	�D
(conv2d_18_conv2d_readvariableop_resource:��8
)conv2d_18_biasadd_readvariableop_resource:	�D
(conv2d_19_conv2d_readvariableop_resource:��8
)conv2d_19_biasadd_readvariableop_resource:	�D
(conv2d_20_conv2d_readvariableop_resource:��8
)conv2d_20_biasadd_readvariableop_resource:	�D
(conv2d_21_conv2d_readvariableop_resource:��8
)conv2d_21_biasadd_readvariableop_resource:	�D
(conv2d_22_conv2d_readvariableop_resource:��8
)conv2d_22_biasadd_readvariableop_resource:	�W
;conv2d_transpose_3_conv2d_transpose_readvariableop_resource:��A
2conv2d_transpose_3_biasadd_readvariableop_resource:	�D
(conv2d_23_conv2d_readvariableop_resource:��8
)conv2d_23_biasadd_readvariableop_resource:	�D
(conv2d_24_conv2d_readvariableop_resource:��8
)conv2d_24_biasadd_readvariableop_resource:	�W
;conv2d_transpose_4_conv2d_transpose_readvariableop_resource:��A
2conv2d_transpose_4_biasadd_readvariableop_resource:	�D
(conv2d_25_conv2d_readvariableop_resource:��8
)conv2d_25_biasadd_readvariableop_resource:	�D
(conv2d_26_conv2d_readvariableop_resource:��8
)conv2d_26_biasadd_readvariableop_resource:	�V
;conv2d_transpose_5_conv2d_transpose_readvariableop_resource:@�@
2conv2d_transpose_5_biasadd_readvariableop_resource:@C
(conv2d_27_conv2d_readvariableop_resource:�@7
)conv2d_27_biasadd_readvariableop_resource:@B
(conv2d_28_conv2d_readvariableop_resource:@@7
)conv2d_28_biasadd_readvariableop_resource:@B
(conv2d_29_conv2d_readvariableop_resource:@7
)conv2d_29_biasadd_readvariableop_resource:
identity�� conv2d_15/BiasAdd/ReadVariableOp�conv2d_15/Conv2D/ReadVariableOp� conv2d_16/BiasAdd/ReadVariableOp�conv2d_16/Conv2D/ReadVariableOp� conv2d_17/BiasAdd/ReadVariableOp�conv2d_17/Conv2D/ReadVariableOp� conv2d_18/BiasAdd/ReadVariableOp�conv2d_18/Conv2D/ReadVariableOp� conv2d_19/BiasAdd/ReadVariableOp�conv2d_19/Conv2D/ReadVariableOp� conv2d_20/BiasAdd/ReadVariableOp�conv2d_20/Conv2D/ReadVariableOp� conv2d_21/BiasAdd/ReadVariableOp�conv2d_21/Conv2D/ReadVariableOp� conv2d_22/BiasAdd/ReadVariableOp�conv2d_22/Conv2D/ReadVariableOp� conv2d_23/BiasAdd/ReadVariableOp�conv2d_23/Conv2D/ReadVariableOp� conv2d_24/BiasAdd/ReadVariableOp�conv2d_24/Conv2D/ReadVariableOp� conv2d_25/BiasAdd/ReadVariableOp�conv2d_25/Conv2D/ReadVariableOp� conv2d_26/BiasAdd/ReadVariableOp�conv2d_26/Conv2D/ReadVariableOp� conv2d_27/BiasAdd/ReadVariableOp�conv2d_27/Conv2D/ReadVariableOp� conv2d_28/BiasAdd/ReadVariableOp�conv2d_28/Conv2D/ReadVariableOp� conv2d_29/BiasAdd/ReadVariableOp�conv2d_29/Conv2D/ReadVariableOp�)conv2d_transpose_3/BiasAdd/ReadVariableOp�2conv2d_transpose_3/conv2d_transpose/ReadVariableOp�)conv2d_transpose_4/BiasAdd/ReadVariableOp�2conv2d_transpose_4/conv2d_transpose/ReadVariableOp�)conv2d_transpose_5/BiasAdd/ReadVariableOp�2conv2d_transpose_5/conv2d_transpose/ReadVariableOp�
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
conv2d_15/Conv2DConv2Dinputs'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@*
paddingSAME*
strides
�
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@l
conv2d_15/ReluReluconv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@@�
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
conv2d_16/Conv2DConv2Dconv2d_15/Relu:activations:0'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@*
paddingSAME*
strides
�
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@l
conv2d_16/ReluReluconv2d_16/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@@�
max_pooling2d_3/MaxPoolMaxPoolconv2d_16/Relu:activations:0*/
_output_shapes
:���������  @*
ksize
*
paddingVALID*
strides
�
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
conv2d_17/Conv2DConv2D max_pooling2d_3/MaxPool:output:0'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
�
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �m
conv2d_17/ReluReluconv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:���������  ��
conv2d_18/Conv2D/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_18/Conv2DConv2Dconv2d_17/Relu:activations:0'conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
�
 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0(conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �m
conv2d_18/ReluReluconv2d_18/BiasAdd:output:0*
T0*0
_output_shapes
:���������  ��
max_pooling2d_4/MaxPoolMaxPoolconv2d_18/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
�
conv2d_19/Conv2D/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_19/Conv2DConv2D max_pooling2d_4/MaxPool:output:0'conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������m
conv2d_19/ReluReluconv2d_19/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_20/Conv2DConv2Dconv2d_19/Relu:activations:0'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������m
conv2d_20/ReluReluconv2d_20/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
max_pooling2d_5/MaxPoolMaxPoolconv2d_20/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
�
conv2d_21/Conv2D/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_21/Conv2DConv2D max_pooling2d_5/MaxPool:output:0'conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
 conv2d_21/BiasAdd/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_21/BiasAddBiasAddconv2d_21/Conv2D:output:0(conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������m
conv2d_21/ReluReluconv2d_21/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_22/Conv2DConv2Dconv2d_21/Relu:activations:0'conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������m
conv2d_22/ReluReluconv2d_22/BiasAdd:output:0*
T0*0
_output_shapes
:����������r
conv2d_transpose_3/ShapeShapeconv2d_22/Relu:activations:0*
T0*
_output_shapes
::��p
&conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_3/strided_sliceStridedSlice!conv2d_transpose_3/Shape:output:0/conv2d_transpose_3/strided_slice/stack:output:01conv2d_transpose_3/strided_slice/stack_1:output:01conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value
B :��
conv2d_transpose_3/stackPack)conv2d_transpose_3/strided_slice:output:0#conv2d_transpose_3/stack/1:output:0#conv2d_transpose_3/stack/2:output:0#conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_3/strided_slice_1StridedSlice!conv2d_transpose_3/stack:output:01conv2d_transpose_3/strided_slice_1/stack:output:03conv2d_transpose_3/strided_slice_1/stack_1:output:03conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype0�
#conv2d_transpose_3/conv2d_transposeConv2DBackpropInput!conv2d_transpose_3/stack:output:0:conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0conv2d_22/Relu:activations:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
)conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_transpose_3/BiasAddBiasAdd,conv2d_transpose_3/conv2d_transpose:output:01conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������[
concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_3/concatConcatV2#conv2d_transpose_3/BiasAdd:output:0conv2d_20/Relu:activations:0"concatenate_3/concat/axis:output:0*
N*
T0*0
_output_shapes
:�����������
conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_23/Conv2DConv2Dconcatenate_3/concat:output:0'conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������m
conv2d_23/ReluReluconv2d_23/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_24/Conv2DConv2Dconv2d_23/Relu:activations:0'conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0(conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������m
conv2d_24/ReluReluconv2d_24/BiasAdd:output:0*
T0*0
_output_shapes
:����������r
conv2d_transpose_4/ShapeShapeconv2d_24/Relu:activations:0*
T0*
_output_shapes
::��p
&conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_4/strided_sliceStridedSlice!conv2d_transpose_4/Shape:output:0/conv2d_transpose_4/strided_slice/stack:output:01conv2d_transpose_4/strided_slice/stack_1:output:01conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B : \
conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B : ]
conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value
B :��
conv2d_transpose_4/stackPack)conv2d_transpose_4/strided_slice:output:0#conv2d_transpose_4/stack/1:output:0#conv2d_transpose_4/stack/2:output:0#conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_4/strided_slice_1StridedSlice!conv2d_transpose_4/stack:output:01conv2d_transpose_4/strided_slice_1/stack:output:03conv2d_transpose_4/strided_slice_1/stack_1:output:03conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_4_conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype0�
#conv2d_transpose_4/conv2d_transposeConv2DBackpropInput!conv2d_transpose_4/stack:output:0:conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0conv2d_24/Relu:activations:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
�
)conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_transpose_4/BiasAddBiasAdd,conv2d_transpose_4/conv2d_transpose:output:01conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �[
concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_4/concatConcatV2#conv2d_transpose_4/BiasAdd:output:0conv2d_18/Relu:activations:0"concatenate_4/concat/axis:output:0*
N*
T0*0
_output_shapes
:���������  ��
conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_25/Conv2DConv2Dconcatenate_4/concat:output:0'conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
�
 conv2d_25/BiasAdd/ReadVariableOpReadVariableOp)conv2d_25_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_25/BiasAddBiasAddconv2d_25/Conv2D:output:0(conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �m
conv2d_25/ReluReluconv2d_25/BiasAdd:output:0*
T0*0
_output_shapes
:���������  ��
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_26/Conv2DConv2Dconv2d_25/Relu:activations:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
�
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �m
conv2d_26/ReluReluconv2d_26/BiasAdd:output:0*
T0*0
_output_shapes
:���������  �r
conv2d_transpose_5/ShapeShapeconv2d_26/Relu:activations:0*
T0*
_output_shapes
::��p
&conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_5/strided_sliceStridedSlice!conv2d_transpose_5/Shape:output:0/conv2d_transpose_5/strided_slice/stack:output:01conv2d_transpose_5/strided_slice/stack_1:output:01conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@\
conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@\
conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@�
conv2d_transpose_5/stackPack)conv2d_transpose_5/strided_slice:output:0#conv2d_transpose_5/stack/1:output:0#conv2d_transpose_5/stack/2:output:0#conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_5/strided_slice_1StridedSlice!conv2d_transpose_5/stack:output:01conv2d_transpose_5/strided_slice_1/stack:output:03conv2d_transpose_5/strided_slice_1/stack_1:output:03conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_5_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
#conv2d_transpose_5/conv2d_transposeConv2DBackpropInput!conv2d_transpose_5/stack:output:0:conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0conv2d_26/Relu:activations:0*
T0*/
_output_shapes
:���������@@@*
paddingSAME*
strides
�
)conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_transpose_5/BiasAddBiasAdd,conv2d_transpose_5/conv2d_transpose:output:01conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@[
concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_5/concatConcatV2#conv2d_transpose_5/BiasAdd:output:0conv2d_16/Relu:activations:0"concatenate_5/concat/axis:output:0*
N*
T0*0
_output_shapes
:���������@@��
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*'
_output_shapes
:�@*
dtype0�
conv2d_27/Conv2DConv2Dconcatenate_5/concat:output:0'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@*
paddingSAME*
strides
�
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@l
conv2d_27/ReluReluconv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@@�
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
conv2d_28/Conv2DConv2Dconv2d_27/Relu:activations:0'conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@*
paddingSAME*
strides
�
 conv2d_28/BiasAdd/ReadVariableOpReadVariableOp)conv2d_28_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_28/BiasAddBiasAddconv2d_28/Conv2D:output:0(conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@l
conv2d_28/ReluReluconv2d_28/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@@�
conv2d_29/Conv2D/ReadVariableOpReadVariableOp(conv2d_29_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
conv2d_29/Conv2DConv2Dconv2d_28/Relu:activations:0'conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingVALID*
strides
�
 conv2d_29/BiasAdd/ReadVariableOpReadVariableOp)conv2d_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_29/BiasAddBiasAddconv2d_29/Conv2D:output:0(conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@r
conv2d_29/SigmoidSigmoidconv2d_29/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@l
IdentityIdentityconv2d_29/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:���������@@�

NoOpNoOp!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp!^conv2d_18/BiasAdd/ReadVariableOp ^conv2d_18/Conv2D/ReadVariableOp!^conv2d_19/BiasAdd/ReadVariableOp ^conv2d_19/Conv2D/ReadVariableOp!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp!^conv2d_21/BiasAdd/ReadVariableOp ^conv2d_21/Conv2D/ReadVariableOp!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp!^conv2d_23/BiasAdd/ReadVariableOp ^conv2d_23/Conv2D/ReadVariableOp!^conv2d_24/BiasAdd/ReadVariableOp ^conv2d_24/Conv2D/ReadVariableOp!^conv2d_25/BiasAdd/ReadVariableOp ^conv2d_25/Conv2D/ReadVariableOp!^conv2d_26/BiasAdd/ReadVariableOp ^conv2d_26/Conv2D/ReadVariableOp!^conv2d_27/BiasAdd/ReadVariableOp ^conv2d_27/Conv2D/ReadVariableOp!^conv2d_28/BiasAdd/ReadVariableOp ^conv2d_28/Conv2D/ReadVariableOp!^conv2d_29/BiasAdd/ReadVariableOp ^conv2d_29/Conv2D/ReadVariableOp*^conv2d_transpose_3/BiasAdd/ReadVariableOp3^conv2d_transpose_3/conv2d_transpose/ReadVariableOp*^conv2d_transpose_4/BiasAdd/ReadVariableOp3^conv2d_transpose_4/conv2d_transpose/ReadVariableOp*^conv2d_transpose_5/BiasAdd/ReadVariableOp3^conv2d_transpose_5/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:���������@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp2D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp2D
 conv2d_18/BiasAdd/ReadVariableOp conv2d_18/BiasAdd/ReadVariableOp2B
conv2d_18/Conv2D/ReadVariableOpconv2d_18/Conv2D/ReadVariableOp2D
 conv2d_19/BiasAdd/ReadVariableOp conv2d_19/BiasAdd/ReadVariableOp2B
conv2d_19/Conv2D/ReadVariableOpconv2d_19/Conv2D/ReadVariableOp2D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp2D
 conv2d_21/BiasAdd/ReadVariableOp conv2d_21/BiasAdd/ReadVariableOp2B
conv2d_21/Conv2D/ReadVariableOpconv2d_21/Conv2D/ReadVariableOp2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp2D
 conv2d_23/BiasAdd/ReadVariableOp conv2d_23/BiasAdd/ReadVariableOp2B
conv2d_23/Conv2D/ReadVariableOpconv2d_23/Conv2D/ReadVariableOp2D
 conv2d_24/BiasAdd/ReadVariableOp conv2d_24/BiasAdd/ReadVariableOp2B
conv2d_24/Conv2D/ReadVariableOpconv2d_24/Conv2D/ReadVariableOp2D
 conv2d_25/BiasAdd/ReadVariableOp conv2d_25/BiasAdd/ReadVariableOp2B
conv2d_25/Conv2D/ReadVariableOpconv2d_25/Conv2D/ReadVariableOp2D
 conv2d_26/BiasAdd/ReadVariableOp conv2d_26/BiasAdd/ReadVariableOp2B
conv2d_26/Conv2D/ReadVariableOpconv2d_26/Conv2D/ReadVariableOp2D
 conv2d_27/BiasAdd/ReadVariableOp conv2d_27/BiasAdd/ReadVariableOp2B
conv2d_27/Conv2D/ReadVariableOpconv2d_27/Conv2D/ReadVariableOp2D
 conv2d_28/BiasAdd/ReadVariableOp conv2d_28/BiasAdd/ReadVariableOp2B
conv2d_28/Conv2D/ReadVariableOpconv2d_28/Conv2D/ReadVariableOp2D
 conv2d_29/BiasAdd/ReadVariableOp conv2d_29/BiasAdd/ReadVariableOp2B
conv2d_29/Conv2D/ReadVariableOpconv2d_29/Conv2D/ReadVariableOp2V
)conv2d_transpose_3/BiasAdd/ReadVariableOp)conv2d_transpose_3/BiasAdd/ReadVariableOp2h
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_4/BiasAdd/ReadVariableOp)conv2d_transpose_4/BiasAdd/ReadVariableOp2h
2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_5/BiasAdd/ReadVariableOp)conv2d_transpose_5/BiasAdd/ReadVariableOp2h
2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2conv2d_transpose_5/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
L
0__inference_max_pooling2d_5_layer_call_fn_308842

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_306889�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
۳
�m
__inference__traced_save_309899
file_prefixA
'read_disablecopyonread_conv2d_15_kernel:@5
'read_1_disablecopyonread_conv2d_15_bias:@C
)read_2_disablecopyonread_conv2d_16_kernel:@@5
'read_3_disablecopyonread_conv2d_16_bias:@D
)read_4_disablecopyonread_conv2d_17_kernel:@�6
'read_5_disablecopyonread_conv2d_17_bias:	�E
)read_6_disablecopyonread_conv2d_18_kernel:��6
'read_7_disablecopyonread_conv2d_18_bias:	�E
)read_8_disablecopyonread_conv2d_19_kernel:��6
'read_9_disablecopyonread_conv2d_19_bias:	�F
*read_10_disablecopyonread_conv2d_20_kernel:��7
(read_11_disablecopyonread_conv2d_20_bias:	�F
*read_12_disablecopyonread_conv2d_21_kernel:��7
(read_13_disablecopyonread_conv2d_21_bias:	�F
*read_14_disablecopyonread_conv2d_22_kernel:��7
(read_15_disablecopyonread_conv2d_22_bias:	�O
3read_16_disablecopyonread_conv2d_transpose_3_kernel:��@
1read_17_disablecopyonread_conv2d_transpose_3_bias:	�F
*read_18_disablecopyonread_conv2d_23_kernel:��7
(read_19_disablecopyonread_conv2d_23_bias:	�F
*read_20_disablecopyonread_conv2d_24_kernel:��7
(read_21_disablecopyonread_conv2d_24_bias:	�O
3read_22_disablecopyonread_conv2d_transpose_4_kernel:��@
1read_23_disablecopyonread_conv2d_transpose_4_bias:	�F
*read_24_disablecopyonread_conv2d_25_kernel:��7
(read_25_disablecopyonread_conv2d_25_bias:	�F
*read_26_disablecopyonread_conv2d_26_kernel:��7
(read_27_disablecopyonread_conv2d_26_bias:	�N
3read_28_disablecopyonread_conv2d_transpose_5_kernel:@�?
1read_29_disablecopyonread_conv2d_transpose_5_bias:@E
*read_30_disablecopyonread_conv2d_27_kernel:�@6
(read_31_disablecopyonread_conv2d_27_bias:@D
*read_32_disablecopyonread_conv2d_28_kernel:@@6
(read_33_disablecopyonread_conv2d_28_bias:@D
*read_34_disablecopyonread_conv2d_29_kernel:@6
(read_35_disablecopyonread_conv2d_29_bias:-
#read_36_disablecopyonread_iteration:	 1
'read_37_disablecopyonread_learning_rate: K
1read_38_disablecopyonread_adam_m_conv2d_15_kernel:@K
1read_39_disablecopyonread_adam_v_conv2d_15_kernel:@=
/read_40_disablecopyonread_adam_m_conv2d_15_bias:@=
/read_41_disablecopyonread_adam_v_conv2d_15_bias:@K
1read_42_disablecopyonread_adam_m_conv2d_16_kernel:@@K
1read_43_disablecopyonread_adam_v_conv2d_16_kernel:@@=
/read_44_disablecopyonread_adam_m_conv2d_16_bias:@=
/read_45_disablecopyonread_adam_v_conv2d_16_bias:@L
1read_46_disablecopyonread_adam_m_conv2d_17_kernel:@�L
1read_47_disablecopyonread_adam_v_conv2d_17_kernel:@�>
/read_48_disablecopyonread_adam_m_conv2d_17_bias:	�>
/read_49_disablecopyonread_adam_v_conv2d_17_bias:	�M
1read_50_disablecopyonread_adam_m_conv2d_18_kernel:��M
1read_51_disablecopyonread_adam_v_conv2d_18_kernel:��>
/read_52_disablecopyonread_adam_m_conv2d_18_bias:	�>
/read_53_disablecopyonread_adam_v_conv2d_18_bias:	�M
1read_54_disablecopyonread_adam_m_conv2d_19_kernel:��M
1read_55_disablecopyonread_adam_v_conv2d_19_kernel:��>
/read_56_disablecopyonread_adam_m_conv2d_19_bias:	�>
/read_57_disablecopyonread_adam_v_conv2d_19_bias:	�M
1read_58_disablecopyonread_adam_m_conv2d_20_kernel:��M
1read_59_disablecopyonread_adam_v_conv2d_20_kernel:��>
/read_60_disablecopyonread_adam_m_conv2d_20_bias:	�>
/read_61_disablecopyonread_adam_v_conv2d_20_bias:	�M
1read_62_disablecopyonread_adam_m_conv2d_21_kernel:��M
1read_63_disablecopyonread_adam_v_conv2d_21_kernel:��>
/read_64_disablecopyonread_adam_m_conv2d_21_bias:	�>
/read_65_disablecopyonread_adam_v_conv2d_21_bias:	�M
1read_66_disablecopyonread_adam_m_conv2d_22_kernel:��M
1read_67_disablecopyonread_adam_v_conv2d_22_kernel:��>
/read_68_disablecopyonread_adam_m_conv2d_22_bias:	�>
/read_69_disablecopyonread_adam_v_conv2d_22_bias:	�V
:read_70_disablecopyonread_adam_m_conv2d_transpose_3_kernel:��V
:read_71_disablecopyonread_adam_v_conv2d_transpose_3_kernel:��G
8read_72_disablecopyonread_adam_m_conv2d_transpose_3_bias:	�G
8read_73_disablecopyonread_adam_v_conv2d_transpose_3_bias:	�M
1read_74_disablecopyonread_adam_m_conv2d_23_kernel:��M
1read_75_disablecopyonread_adam_v_conv2d_23_kernel:��>
/read_76_disablecopyonread_adam_m_conv2d_23_bias:	�>
/read_77_disablecopyonread_adam_v_conv2d_23_bias:	�M
1read_78_disablecopyonread_adam_m_conv2d_24_kernel:��M
1read_79_disablecopyonread_adam_v_conv2d_24_kernel:��>
/read_80_disablecopyonread_adam_m_conv2d_24_bias:	�>
/read_81_disablecopyonread_adam_v_conv2d_24_bias:	�V
:read_82_disablecopyonread_adam_m_conv2d_transpose_4_kernel:��V
:read_83_disablecopyonread_adam_v_conv2d_transpose_4_kernel:��G
8read_84_disablecopyonread_adam_m_conv2d_transpose_4_bias:	�G
8read_85_disablecopyonread_adam_v_conv2d_transpose_4_bias:	�M
1read_86_disablecopyonread_adam_m_conv2d_25_kernel:��M
1read_87_disablecopyonread_adam_v_conv2d_25_kernel:��>
/read_88_disablecopyonread_adam_m_conv2d_25_bias:	�>
/read_89_disablecopyonread_adam_v_conv2d_25_bias:	�M
1read_90_disablecopyonread_adam_m_conv2d_26_kernel:��M
1read_91_disablecopyonread_adam_v_conv2d_26_kernel:��>
/read_92_disablecopyonread_adam_m_conv2d_26_bias:	�>
/read_93_disablecopyonread_adam_v_conv2d_26_bias:	�U
:read_94_disablecopyonread_adam_m_conv2d_transpose_5_kernel:@�U
:read_95_disablecopyonread_adam_v_conv2d_transpose_5_kernel:@�F
8read_96_disablecopyonread_adam_m_conv2d_transpose_5_bias:@F
8read_97_disablecopyonread_adam_v_conv2d_transpose_5_bias:@L
1read_98_disablecopyonread_adam_m_conv2d_27_kernel:�@L
1read_99_disablecopyonread_adam_v_conv2d_27_kernel:�@>
0read_100_disablecopyonread_adam_m_conv2d_27_bias:@>
0read_101_disablecopyonread_adam_v_conv2d_27_bias:@L
2read_102_disablecopyonread_adam_m_conv2d_28_kernel:@@L
2read_103_disablecopyonread_adam_v_conv2d_28_kernel:@@>
0read_104_disablecopyonread_adam_m_conv2d_28_bias:@>
0read_105_disablecopyonread_adam_v_conv2d_28_bias:@L
2read_106_disablecopyonread_adam_m_conv2d_29_kernel:@L
2read_107_disablecopyonread_adam_v_conv2d_29_kernel:@>
0read_108_disablecopyonread_adam_m_conv2d_29_bias:>
0read_109_disablecopyonread_adam_v_conv2d_29_bias:,
"read_110_disablecopyonread_total_1: ,
"read_111_disablecopyonread_count_1: *
 read_112_disablecopyonread_total: *
 read_113_disablecopyonread_count: 
savev2_const
identity_229��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_100/DisableCopyOnRead�Read_100/ReadVariableOp�Read_101/DisableCopyOnRead�Read_101/ReadVariableOp�Read_102/DisableCopyOnRead�Read_102/ReadVariableOp�Read_103/DisableCopyOnRead�Read_103/ReadVariableOp�Read_104/DisableCopyOnRead�Read_104/ReadVariableOp�Read_105/DisableCopyOnRead�Read_105/ReadVariableOp�Read_106/DisableCopyOnRead�Read_106/ReadVariableOp�Read_107/DisableCopyOnRead�Read_107/ReadVariableOp�Read_108/DisableCopyOnRead�Read_108/ReadVariableOp�Read_109/DisableCopyOnRead�Read_109/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_110/DisableCopyOnRead�Read_110/ReadVariableOp�Read_111/DisableCopyOnRead�Read_111/ReadVariableOp�Read_112/DisableCopyOnRead�Read_112/ReadVariableOp�Read_113/DisableCopyOnRead�Read_113/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_52/DisableCopyOnRead�Read_52/ReadVariableOp�Read_53/DisableCopyOnRead�Read_53/ReadVariableOp�Read_54/DisableCopyOnRead�Read_54/ReadVariableOp�Read_55/DisableCopyOnRead�Read_55/ReadVariableOp�Read_56/DisableCopyOnRead�Read_56/ReadVariableOp�Read_57/DisableCopyOnRead�Read_57/ReadVariableOp�Read_58/DisableCopyOnRead�Read_58/ReadVariableOp�Read_59/DisableCopyOnRead�Read_59/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_60/DisableCopyOnRead�Read_60/ReadVariableOp�Read_61/DisableCopyOnRead�Read_61/ReadVariableOp�Read_62/DisableCopyOnRead�Read_62/ReadVariableOp�Read_63/DisableCopyOnRead�Read_63/ReadVariableOp�Read_64/DisableCopyOnRead�Read_64/ReadVariableOp�Read_65/DisableCopyOnRead�Read_65/ReadVariableOp�Read_66/DisableCopyOnRead�Read_66/ReadVariableOp�Read_67/DisableCopyOnRead�Read_67/ReadVariableOp�Read_68/DisableCopyOnRead�Read_68/ReadVariableOp�Read_69/DisableCopyOnRead�Read_69/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_70/DisableCopyOnRead�Read_70/ReadVariableOp�Read_71/DisableCopyOnRead�Read_71/ReadVariableOp�Read_72/DisableCopyOnRead�Read_72/ReadVariableOp�Read_73/DisableCopyOnRead�Read_73/ReadVariableOp�Read_74/DisableCopyOnRead�Read_74/ReadVariableOp�Read_75/DisableCopyOnRead�Read_75/ReadVariableOp�Read_76/DisableCopyOnRead�Read_76/ReadVariableOp�Read_77/DisableCopyOnRead�Read_77/ReadVariableOp�Read_78/DisableCopyOnRead�Read_78/ReadVariableOp�Read_79/DisableCopyOnRead�Read_79/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_80/DisableCopyOnRead�Read_80/ReadVariableOp�Read_81/DisableCopyOnRead�Read_81/ReadVariableOp�Read_82/DisableCopyOnRead�Read_82/ReadVariableOp�Read_83/DisableCopyOnRead�Read_83/ReadVariableOp�Read_84/DisableCopyOnRead�Read_84/ReadVariableOp�Read_85/DisableCopyOnRead�Read_85/ReadVariableOp�Read_86/DisableCopyOnRead�Read_86/ReadVariableOp�Read_87/DisableCopyOnRead�Read_87/ReadVariableOp�Read_88/DisableCopyOnRead�Read_88/ReadVariableOp�Read_89/DisableCopyOnRead�Read_89/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOp�Read_90/DisableCopyOnRead�Read_90/ReadVariableOp�Read_91/DisableCopyOnRead�Read_91/ReadVariableOp�Read_92/DisableCopyOnRead�Read_92/ReadVariableOp�Read_93/DisableCopyOnRead�Read_93/ReadVariableOp�Read_94/DisableCopyOnRead�Read_94/ReadVariableOp�Read_95/DisableCopyOnRead�Read_95/ReadVariableOp�Read_96/DisableCopyOnRead�Read_96/ReadVariableOp�Read_97/DisableCopyOnRead�Read_97/ReadVariableOp�Read_98/DisableCopyOnRead�Read_98/ReadVariableOp�Read_99/DisableCopyOnRead�Read_99/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: y
Read/DisableCopyOnReadDisableCopyOnRead'read_disablecopyonread_conv2d_15_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp'read_disablecopyonread_conv2d_15_kernel^Read/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0q
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
:@{
Read_1/DisableCopyOnReadDisableCopyOnRead'read_1_disablecopyonread_conv2d_15_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp'read_1_disablecopyonread_conv2d_15_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:@}
Read_2/DisableCopyOnReadDisableCopyOnRead)read_2_disablecopyonread_conv2d_16_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp)read_2_disablecopyonread_conv2d_16_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0u

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@k

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@{
Read_3/DisableCopyOnReadDisableCopyOnRead'read_3_disablecopyonread_conv2d_16_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp'read_3_disablecopyonread_conv2d_16_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:@}
Read_4/DisableCopyOnReadDisableCopyOnRead)read_4_disablecopyonread_conv2d_17_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp)read_4_disablecopyonread_conv2d_17_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@�*
dtype0v

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@�l

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*'
_output_shapes
:@�{
Read_5/DisableCopyOnReadDisableCopyOnRead'read_5_disablecopyonread_conv2d_17_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp'read_5_disablecopyonread_conv2d_17_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes	
:�}
Read_6/DisableCopyOnReadDisableCopyOnRead)read_6_disablecopyonread_conv2d_18_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp)read_6_disablecopyonread_conv2d_18_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0x
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*(
_output_shapes
:��{
Read_7/DisableCopyOnReadDisableCopyOnRead'read_7_disablecopyonread_conv2d_18_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp'read_7_disablecopyonread_conv2d_18_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes	
:�}
Read_8/DisableCopyOnReadDisableCopyOnRead)read_8_disablecopyonread_conv2d_19_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp)read_8_disablecopyonread_conv2d_19_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0x
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*(
_output_shapes
:��{
Read_9/DisableCopyOnReadDisableCopyOnRead'read_9_disablecopyonread_conv2d_19_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp'read_9_disablecopyonread_conv2d_19_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes	
:�
Read_10/DisableCopyOnReadDisableCopyOnRead*read_10_disablecopyonread_conv2d_20_kernel"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp*read_10_disablecopyonread_conv2d_20_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*(
_output_shapes
:��}
Read_11/DisableCopyOnReadDisableCopyOnRead(read_11_disablecopyonread_conv2d_20_bias"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp(read_11_disablecopyonread_conv2d_20_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes	
:�
Read_12/DisableCopyOnReadDisableCopyOnRead*read_12_disablecopyonread_conv2d_21_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp*read_12_disablecopyonread_conv2d_21_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*(
_output_shapes
:��}
Read_13/DisableCopyOnReadDisableCopyOnRead(read_13_disablecopyonread_conv2d_21_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp(read_13_disablecopyonread_conv2d_21_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes	
:�
Read_14/DisableCopyOnReadDisableCopyOnRead*read_14_disablecopyonread_conv2d_22_kernel"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp*read_14_disablecopyonread_conv2d_22_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*(
_output_shapes
:��}
Read_15/DisableCopyOnReadDisableCopyOnRead(read_15_disablecopyonread_conv2d_22_bias"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp(read_15_disablecopyonread_conv2d_22_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_16/DisableCopyOnReadDisableCopyOnRead3read_16_disablecopyonread_conv2d_transpose_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp3read_16_disablecopyonread_conv2d_transpose_3_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_17/DisableCopyOnReadDisableCopyOnRead1read_17_disablecopyonread_conv2d_transpose_3_bias"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp1read_17_disablecopyonread_conv2d_transpose_3_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes	
:�
Read_18/DisableCopyOnReadDisableCopyOnRead*read_18_disablecopyonread_conv2d_23_kernel"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp*read_18_disablecopyonread_conv2d_23_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*(
_output_shapes
:��}
Read_19/DisableCopyOnReadDisableCopyOnRead(read_19_disablecopyonread_conv2d_23_bias"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp(read_19_disablecopyonread_conv2d_23_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes	
:�
Read_20/DisableCopyOnReadDisableCopyOnRead*read_20_disablecopyonread_conv2d_24_kernel"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp*read_20_disablecopyonread_conv2d_24_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*(
_output_shapes
:��}
Read_21/DisableCopyOnReadDisableCopyOnRead(read_21_disablecopyonread_conv2d_24_bias"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp(read_21_disablecopyonread_conv2d_24_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_22/DisableCopyOnReadDisableCopyOnRead3read_22_disablecopyonread_conv2d_transpose_4_kernel"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp3read_22_disablecopyonread_conv2d_transpose_4_kernel^Read_22/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_23/DisableCopyOnReadDisableCopyOnRead1read_23_disablecopyonread_conv2d_transpose_4_bias"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp1read_23_disablecopyonread_conv2d_transpose_4_bias^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes	
:�
Read_24/DisableCopyOnReadDisableCopyOnRead*read_24_disablecopyonread_conv2d_25_kernel"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp*read_24_disablecopyonread_conv2d_25_kernel^Read_24/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*(
_output_shapes
:��}
Read_25/DisableCopyOnReadDisableCopyOnRead(read_25_disablecopyonread_conv2d_25_bias"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp(read_25_disablecopyonread_conv2d_25_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes	
:�
Read_26/DisableCopyOnReadDisableCopyOnRead*read_26_disablecopyonread_conv2d_26_kernel"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp*read_26_disablecopyonread_conv2d_26_kernel^Read_26/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*(
_output_shapes
:��}
Read_27/DisableCopyOnReadDisableCopyOnRead(read_27_disablecopyonread_conv2d_26_bias"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp(read_27_disablecopyonread_conv2d_26_bias^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_28/DisableCopyOnReadDisableCopyOnRead3read_28_disablecopyonread_conv2d_transpose_5_kernel"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp3read_28_disablecopyonread_conv2d_transpose_5_kernel^Read_28/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@�*
dtype0x
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@�n
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*'
_output_shapes
:@��
Read_29/DisableCopyOnReadDisableCopyOnRead1read_29_disablecopyonread_conv2d_transpose_5_bias"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp1read_29_disablecopyonread_conv2d_transpose_5_bias^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_30/DisableCopyOnReadDisableCopyOnRead*read_30_disablecopyonread_conv2d_27_kernel"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp*read_30_disablecopyonread_conv2d_27_kernel^Read_30/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:�@*
dtype0x
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:�@n
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*'
_output_shapes
:�@}
Read_31/DisableCopyOnReadDisableCopyOnRead(read_31_disablecopyonread_conv2d_27_bias"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp(read_31_disablecopyonread_conv2d_27_bias^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_32/DisableCopyOnReadDisableCopyOnRead*read_32_disablecopyonread_conv2d_28_kernel"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp*read_32_disablecopyonread_conv2d_28_kernel^Read_32/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0w
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@m
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@}
Read_33/DisableCopyOnReadDisableCopyOnRead(read_33_disablecopyonread_conv2d_28_bias"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp(read_33_disablecopyonread_conv2d_28_bias^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_34/DisableCopyOnReadDisableCopyOnRead*read_34_disablecopyonread_conv2d_29_kernel"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp*read_34_disablecopyonread_conv2d_29_kernel^Read_34/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0w
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@m
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*&
_output_shapes
:@}
Read_35/DisableCopyOnReadDisableCopyOnRead(read_35_disablecopyonread_conv2d_29_bias"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp(read_35_disablecopyonread_conv2d_29_bias^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_36/DisableCopyOnReadDisableCopyOnRead#read_36_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp#read_36_disablecopyonread_iteration^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_37/DisableCopyOnReadDisableCopyOnRead'read_37_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp'read_37_disablecopyonread_learning_rate^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_38/DisableCopyOnReadDisableCopyOnRead1read_38_disablecopyonread_adam_m_conv2d_15_kernel"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp1read_38_disablecopyonread_adam_m_conv2d_15_kernel^Read_38/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0w
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@m
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*&
_output_shapes
:@�
Read_39/DisableCopyOnReadDisableCopyOnRead1read_39_disablecopyonread_adam_v_conv2d_15_kernel"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOp1read_39_disablecopyonread_adam_v_conv2d_15_kernel^Read_39/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0w
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@m
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*&
_output_shapes
:@�
Read_40/DisableCopyOnReadDisableCopyOnRead/read_40_disablecopyonread_adam_m_conv2d_15_bias"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOp/read_40_disablecopyonread_adam_m_conv2d_15_bias^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_41/DisableCopyOnReadDisableCopyOnRead/read_41_disablecopyonread_adam_v_conv2d_15_bias"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp/read_41_disablecopyonread_adam_v_conv2d_15_bias^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_42/DisableCopyOnReadDisableCopyOnRead1read_42_disablecopyonread_adam_m_conv2d_16_kernel"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOp1read_42_disablecopyonread_adam_m_conv2d_16_kernel^Read_42/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0w
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@m
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@�
Read_43/DisableCopyOnReadDisableCopyOnRead1read_43_disablecopyonread_adam_v_conv2d_16_kernel"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOp1read_43_disablecopyonread_adam_v_conv2d_16_kernel^Read_43/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0w
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@m
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@�
Read_44/DisableCopyOnReadDisableCopyOnRead/read_44_disablecopyonread_adam_m_conv2d_16_bias"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOp/read_44_disablecopyonread_adam_m_conv2d_16_bias^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_45/DisableCopyOnReadDisableCopyOnRead/read_45_disablecopyonread_adam_v_conv2d_16_bias"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOp/read_45_disablecopyonread_adam_v_conv2d_16_bias^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_46/DisableCopyOnReadDisableCopyOnRead1read_46_disablecopyonread_adam_m_conv2d_17_kernel"/device:CPU:0*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOp1read_46_disablecopyonread_adam_m_conv2d_17_kernel^Read_46/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@�*
dtype0x
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@�n
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*'
_output_shapes
:@��
Read_47/DisableCopyOnReadDisableCopyOnRead1read_47_disablecopyonread_adam_v_conv2d_17_kernel"/device:CPU:0*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOp1read_47_disablecopyonread_adam_v_conv2d_17_kernel^Read_47/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@�*
dtype0x
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@�n
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*'
_output_shapes
:@��
Read_48/DisableCopyOnReadDisableCopyOnRead/read_48_disablecopyonread_adam_m_conv2d_17_bias"/device:CPU:0*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOp/read_48_disablecopyonread_adam_m_conv2d_17_bias^Read_48/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_49/DisableCopyOnReadDisableCopyOnRead/read_49_disablecopyonread_adam_v_conv2d_17_bias"/device:CPU:0*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOp/read_49_disablecopyonread_adam_v_conv2d_17_bias^Read_49/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_50/DisableCopyOnReadDisableCopyOnRead1read_50_disablecopyonread_adam_m_conv2d_18_kernel"/device:CPU:0*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOp1read_50_disablecopyonread_adam_m_conv2d_18_kernel^Read_50/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_51/DisableCopyOnReadDisableCopyOnRead1read_51_disablecopyonread_adam_v_conv2d_18_kernel"/device:CPU:0*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOp1read_51_disablecopyonread_adam_v_conv2d_18_kernel^Read_51/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_52/DisableCopyOnReadDisableCopyOnRead/read_52_disablecopyonread_adam_m_conv2d_18_bias"/device:CPU:0*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOp/read_52_disablecopyonread_adam_m_conv2d_18_bias^Read_52/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_53/DisableCopyOnReadDisableCopyOnRead/read_53_disablecopyonread_adam_v_conv2d_18_bias"/device:CPU:0*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOp/read_53_disablecopyonread_adam_v_conv2d_18_bias^Read_53/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_54/DisableCopyOnReadDisableCopyOnRead1read_54_disablecopyonread_adam_m_conv2d_19_kernel"/device:CPU:0*
_output_shapes
 �
Read_54/ReadVariableOpReadVariableOp1read_54_disablecopyonread_adam_m_conv2d_19_kernel^Read_54/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_55/DisableCopyOnReadDisableCopyOnRead1read_55_disablecopyonread_adam_v_conv2d_19_kernel"/device:CPU:0*
_output_shapes
 �
Read_55/ReadVariableOpReadVariableOp1read_55_disablecopyonread_adam_v_conv2d_19_kernel^Read_55/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_56/DisableCopyOnReadDisableCopyOnRead/read_56_disablecopyonread_adam_m_conv2d_19_bias"/device:CPU:0*
_output_shapes
 �
Read_56/ReadVariableOpReadVariableOp/read_56_disablecopyonread_adam_m_conv2d_19_bias^Read_56/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_112IdentityRead_56/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_57/DisableCopyOnReadDisableCopyOnRead/read_57_disablecopyonread_adam_v_conv2d_19_bias"/device:CPU:0*
_output_shapes
 �
Read_57/ReadVariableOpReadVariableOp/read_57_disablecopyonread_adam_v_conv2d_19_bias^Read_57/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_114IdentityRead_57/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_58/DisableCopyOnReadDisableCopyOnRead1read_58_disablecopyonread_adam_m_conv2d_20_kernel"/device:CPU:0*
_output_shapes
 �
Read_58/ReadVariableOpReadVariableOp1read_58_disablecopyonread_adam_m_conv2d_20_kernel^Read_58/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_116IdentityRead_58/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_117IdentityIdentity_116:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_59/DisableCopyOnReadDisableCopyOnRead1read_59_disablecopyonread_adam_v_conv2d_20_kernel"/device:CPU:0*
_output_shapes
 �
Read_59/ReadVariableOpReadVariableOp1read_59_disablecopyonread_adam_v_conv2d_20_kernel^Read_59/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_118IdentityRead_59/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_119IdentityIdentity_118:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_60/DisableCopyOnReadDisableCopyOnRead/read_60_disablecopyonread_adam_m_conv2d_20_bias"/device:CPU:0*
_output_shapes
 �
Read_60/ReadVariableOpReadVariableOp/read_60_disablecopyonread_adam_m_conv2d_20_bias^Read_60/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_120IdentityRead_60/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_121IdentityIdentity_120:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_61/DisableCopyOnReadDisableCopyOnRead/read_61_disablecopyonread_adam_v_conv2d_20_bias"/device:CPU:0*
_output_shapes
 �
Read_61/ReadVariableOpReadVariableOp/read_61_disablecopyonread_adam_v_conv2d_20_bias^Read_61/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_122IdentityRead_61/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_123IdentityIdentity_122:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_62/DisableCopyOnReadDisableCopyOnRead1read_62_disablecopyonread_adam_m_conv2d_21_kernel"/device:CPU:0*
_output_shapes
 �
Read_62/ReadVariableOpReadVariableOp1read_62_disablecopyonread_adam_m_conv2d_21_kernel^Read_62/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_124IdentityRead_62/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_125IdentityIdentity_124:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_63/DisableCopyOnReadDisableCopyOnRead1read_63_disablecopyonread_adam_v_conv2d_21_kernel"/device:CPU:0*
_output_shapes
 �
Read_63/ReadVariableOpReadVariableOp1read_63_disablecopyonread_adam_v_conv2d_21_kernel^Read_63/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_126IdentityRead_63/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_127IdentityIdentity_126:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_64/DisableCopyOnReadDisableCopyOnRead/read_64_disablecopyonread_adam_m_conv2d_21_bias"/device:CPU:0*
_output_shapes
 �
Read_64/ReadVariableOpReadVariableOp/read_64_disablecopyonread_adam_m_conv2d_21_bias^Read_64/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_128IdentityRead_64/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_129IdentityIdentity_128:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_65/DisableCopyOnReadDisableCopyOnRead/read_65_disablecopyonread_adam_v_conv2d_21_bias"/device:CPU:0*
_output_shapes
 �
Read_65/ReadVariableOpReadVariableOp/read_65_disablecopyonread_adam_v_conv2d_21_bias^Read_65/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_130IdentityRead_65/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_131IdentityIdentity_130:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_66/DisableCopyOnReadDisableCopyOnRead1read_66_disablecopyonread_adam_m_conv2d_22_kernel"/device:CPU:0*
_output_shapes
 �
Read_66/ReadVariableOpReadVariableOp1read_66_disablecopyonread_adam_m_conv2d_22_kernel^Read_66/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_132IdentityRead_66/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_133IdentityIdentity_132:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_67/DisableCopyOnReadDisableCopyOnRead1read_67_disablecopyonread_adam_v_conv2d_22_kernel"/device:CPU:0*
_output_shapes
 �
Read_67/ReadVariableOpReadVariableOp1read_67_disablecopyonread_adam_v_conv2d_22_kernel^Read_67/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_134IdentityRead_67/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_135IdentityIdentity_134:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_68/DisableCopyOnReadDisableCopyOnRead/read_68_disablecopyonread_adam_m_conv2d_22_bias"/device:CPU:0*
_output_shapes
 �
Read_68/ReadVariableOpReadVariableOp/read_68_disablecopyonread_adam_m_conv2d_22_bias^Read_68/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_136IdentityRead_68/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_137IdentityIdentity_136:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_69/DisableCopyOnReadDisableCopyOnRead/read_69_disablecopyonread_adam_v_conv2d_22_bias"/device:CPU:0*
_output_shapes
 �
Read_69/ReadVariableOpReadVariableOp/read_69_disablecopyonread_adam_v_conv2d_22_bias^Read_69/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_138IdentityRead_69/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_139IdentityIdentity_138:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_70/DisableCopyOnReadDisableCopyOnRead:read_70_disablecopyonread_adam_m_conv2d_transpose_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_70/ReadVariableOpReadVariableOp:read_70_disablecopyonread_adam_m_conv2d_transpose_3_kernel^Read_70/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_140IdentityRead_70/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_141IdentityIdentity_140:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_71/DisableCopyOnReadDisableCopyOnRead:read_71_disablecopyonread_adam_v_conv2d_transpose_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_71/ReadVariableOpReadVariableOp:read_71_disablecopyonread_adam_v_conv2d_transpose_3_kernel^Read_71/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_142IdentityRead_71/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_143IdentityIdentity_142:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_72/DisableCopyOnReadDisableCopyOnRead8read_72_disablecopyonread_adam_m_conv2d_transpose_3_bias"/device:CPU:0*
_output_shapes
 �
Read_72/ReadVariableOpReadVariableOp8read_72_disablecopyonread_adam_m_conv2d_transpose_3_bias^Read_72/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_144IdentityRead_72/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_145IdentityIdentity_144:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_73/DisableCopyOnReadDisableCopyOnRead8read_73_disablecopyonread_adam_v_conv2d_transpose_3_bias"/device:CPU:0*
_output_shapes
 �
Read_73/ReadVariableOpReadVariableOp8read_73_disablecopyonread_adam_v_conv2d_transpose_3_bias^Read_73/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_146IdentityRead_73/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_147IdentityIdentity_146:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_74/DisableCopyOnReadDisableCopyOnRead1read_74_disablecopyonread_adam_m_conv2d_23_kernel"/device:CPU:0*
_output_shapes
 �
Read_74/ReadVariableOpReadVariableOp1read_74_disablecopyonread_adam_m_conv2d_23_kernel^Read_74/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_148IdentityRead_74/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_149IdentityIdentity_148:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_75/DisableCopyOnReadDisableCopyOnRead1read_75_disablecopyonread_adam_v_conv2d_23_kernel"/device:CPU:0*
_output_shapes
 �
Read_75/ReadVariableOpReadVariableOp1read_75_disablecopyonread_adam_v_conv2d_23_kernel^Read_75/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_150IdentityRead_75/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_151IdentityIdentity_150:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_76/DisableCopyOnReadDisableCopyOnRead/read_76_disablecopyonread_adam_m_conv2d_23_bias"/device:CPU:0*
_output_shapes
 �
Read_76/ReadVariableOpReadVariableOp/read_76_disablecopyonread_adam_m_conv2d_23_bias^Read_76/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_152IdentityRead_76/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_153IdentityIdentity_152:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_77/DisableCopyOnReadDisableCopyOnRead/read_77_disablecopyonread_adam_v_conv2d_23_bias"/device:CPU:0*
_output_shapes
 �
Read_77/ReadVariableOpReadVariableOp/read_77_disablecopyonread_adam_v_conv2d_23_bias^Read_77/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_154IdentityRead_77/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_155IdentityIdentity_154:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_78/DisableCopyOnReadDisableCopyOnRead1read_78_disablecopyonread_adam_m_conv2d_24_kernel"/device:CPU:0*
_output_shapes
 �
Read_78/ReadVariableOpReadVariableOp1read_78_disablecopyonread_adam_m_conv2d_24_kernel^Read_78/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_156IdentityRead_78/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_157IdentityIdentity_156:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_79/DisableCopyOnReadDisableCopyOnRead1read_79_disablecopyonread_adam_v_conv2d_24_kernel"/device:CPU:0*
_output_shapes
 �
Read_79/ReadVariableOpReadVariableOp1read_79_disablecopyonread_adam_v_conv2d_24_kernel^Read_79/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_158IdentityRead_79/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_159IdentityIdentity_158:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_80/DisableCopyOnReadDisableCopyOnRead/read_80_disablecopyonread_adam_m_conv2d_24_bias"/device:CPU:0*
_output_shapes
 �
Read_80/ReadVariableOpReadVariableOp/read_80_disablecopyonread_adam_m_conv2d_24_bias^Read_80/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_160IdentityRead_80/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_161IdentityIdentity_160:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_81/DisableCopyOnReadDisableCopyOnRead/read_81_disablecopyonread_adam_v_conv2d_24_bias"/device:CPU:0*
_output_shapes
 �
Read_81/ReadVariableOpReadVariableOp/read_81_disablecopyonread_adam_v_conv2d_24_bias^Read_81/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_162IdentityRead_81/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_163IdentityIdentity_162:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_82/DisableCopyOnReadDisableCopyOnRead:read_82_disablecopyonread_adam_m_conv2d_transpose_4_kernel"/device:CPU:0*
_output_shapes
 �
Read_82/ReadVariableOpReadVariableOp:read_82_disablecopyonread_adam_m_conv2d_transpose_4_kernel^Read_82/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_164IdentityRead_82/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_165IdentityIdentity_164:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_83/DisableCopyOnReadDisableCopyOnRead:read_83_disablecopyonread_adam_v_conv2d_transpose_4_kernel"/device:CPU:0*
_output_shapes
 �
Read_83/ReadVariableOpReadVariableOp:read_83_disablecopyonread_adam_v_conv2d_transpose_4_kernel^Read_83/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_166IdentityRead_83/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_167IdentityIdentity_166:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_84/DisableCopyOnReadDisableCopyOnRead8read_84_disablecopyonread_adam_m_conv2d_transpose_4_bias"/device:CPU:0*
_output_shapes
 �
Read_84/ReadVariableOpReadVariableOp8read_84_disablecopyonread_adam_m_conv2d_transpose_4_bias^Read_84/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_168IdentityRead_84/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_169IdentityIdentity_168:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_85/DisableCopyOnReadDisableCopyOnRead8read_85_disablecopyonread_adam_v_conv2d_transpose_4_bias"/device:CPU:0*
_output_shapes
 �
Read_85/ReadVariableOpReadVariableOp8read_85_disablecopyonread_adam_v_conv2d_transpose_4_bias^Read_85/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_170IdentityRead_85/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_171IdentityIdentity_170:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_86/DisableCopyOnReadDisableCopyOnRead1read_86_disablecopyonread_adam_m_conv2d_25_kernel"/device:CPU:0*
_output_shapes
 �
Read_86/ReadVariableOpReadVariableOp1read_86_disablecopyonread_adam_m_conv2d_25_kernel^Read_86/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_172IdentityRead_86/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_173IdentityIdentity_172:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_87/DisableCopyOnReadDisableCopyOnRead1read_87_disablecopyonread_adam_v_conv2d_25_kernel"/device:CPU:0*
_output_shapes
 �
Read_87/ReadVariableOpReadVariableOp1read_87_disablecopyonread_adam_v_conv2d_25_kernel^Read_87/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_174IdentityRead_87/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_175IdentityIdentity_174:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_88/DisableCopyOnReadDisableCopyOnRead/read_88_disablecopyonread_adam_m_conv2d_25_bias"/device:CPU:0*
_output_shapes
 �
Read_88/ReadVariableOpReadVariableOp/read_88_disablecopyonread_adam_m_conv2d_25_bias^Read_88/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_176IdentityRead_88/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_177IdentityIdentity_176:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_89/DisableCopyOnReadDisableCopyOnRead/read_89_disablecopyonread_adam_v_conv2d_25_bias"/device:CPU:0*
_output_shapes
 �
Read_89/ReadVariableOpReadVariableOp/read_89_disablecopyonread_adam_v_conv2d_25_bias^Read_89/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_178IdentityRead_89/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_179IdentityIdentity_178:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_90/DisableCopyOnReadDisableCopyOnRead1read_90_disablecopyonread_adam_m_conv2d_26_kernel"/device:CPU:0*
_output_shapes
 �
Read_90/ReadVariableOpReadVariableOp1read_90_disablecopyonread_adam_m_conv2d_26_kernel^Read_90/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_180IdentityRead_90/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_181IdentityIdentity_180:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_91/DisableCopyOnReadDisableCopyOnRead1read_91_disablecopyonread_adam_v_conv2d_26_kernel"/device:CPU:0*
_output_shapes
 �
Read_91/ReadVariableOpReadVariableOp1read_91_disablecopyonread_adam_v_conv2d_26_kernel^Read_91/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_182IdentityRead_91/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_183IdentityIdentity_182:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_92/DisableCopyOnReadDisableCopyOnRead/read_92_disablecopyonread_adam_m_conv2d_26_bias"/device:CPU:0*
_output_shapes
 �
Read_92/ReadVariableOpReadVariableOp/read_92_disablecopyonread_adam_m_conv2d_26_bias^Read_92/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_184IdentityRead_92/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_185IdentityIdentity_184:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_93/DisableCopyOnReadDisableCopyOnRead/read_93_disablecopyonread_adam_v_conv2d_26_bias"/device:CPU:0*
_output_shapes
 �
Read_93/ReadVariableOpReadVariableOp/read_93_disablecopyonread_adam_v_conv2d_26_bias^Read_93/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_186IdentityRead_93/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_187IdentityIdentity_186:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_94/DisableCopyOnReadDisableCopyOnRead:read_94_disablecopyonread_adam_m_conv2d_transpose_5_kernel"/device:CPU:0*
_output_shapes
 �
Read_94/ReadVariableOpReadVariableOp:read_94_disablecopyonread_adam_m_conv2d_transpose_5_kernel^Read_94/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@�*
dtype0y
Identity_188IdentityRead_94/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@�p
Identity_189IdentityIdentity_188:output:0"/device:CPU:0*
T0*'
_output_shapes
:@��
Read_95/DisableCopyOnReadDisableCopyOnRead:read_95_disablecopyonread_adam_v_conv2d_transpose_5_kernel"/device:CPU:0*
_output_shapes
 �
Read_95/ReadVariableOpReadVariableOp:read_95_disablecopyonread_adam_v_conv2d_transpose_5_kernel^Read_95/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@�*
dtype0y
Identity_190IdentityRead_95/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@�p
Identity_191IdentityIdentity_190:output:0"/device:CPU:0*
T0*'
_output_shapes
:@��
Read_96/DisableCopyOnReadDisableCopyOnRead8read_96_disablecopyonread_adam_m_conv2d_transpose_5_bias"/device:CPU:0*
_output_shapes
 �
Read_96/ReadVariableOpReadVariableOp8read_96_disablecopyonread_adam_m_conv2d_transpose_5_bias^Read_96/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_192IdentityRead_96/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_193IdentityIdentity_192:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_97/DisableCopyOnReadDisableCopyOnRead8read_97_disablecopyonread_adam_v_conv2d_transpose_5_bias"/device:CPU:0*
_output_shapes
 �
Read_97/ReadVariableOpReadVariableOp8read_97_disablecopyonread_adam_v_conv2d_transpose_5_bias^Read_97/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_194IdentityRead_97/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_195IdentityIdentity_194:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_98/DisableCopyOnReadDisableCopyOnRead1read_98_disablecopyonread_adam_m_conv2d_27_kernel"/device:CPU:0*
_output_shapes
 �
Read_98/ReadVariableOpReadVariableOp1read_98_disablecopyonread_adam_m_conv2d_27_kernel^Read_98/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:�@*
dtype0y
Identity_196IdentityRead_98/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:�@p
Identity_197IdentityIdentity_196:output:0"/device:CPU:0*
T0*'
_output_shapes
:�@�
Read_99/DisableCopyOnReadDisableCopyOnRead1read_99_disablecopyonread_adam_v_conv2d_27_kernel"/device:CPU:0*
_output_shapes
 �
Read_99/ReadVariableOpReadVariableOp1read_99_disablecopyonread_adam_v_conv2d_27_kernel^Read_99/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:�@*
dtype0y
Identity_198IdentityRead_99/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:�@p
Identity_199IdentityIdentity_198:output:0"/device:CPU:0*
T0*'
_output_shapes
:�@�
Read_100/DisableCopyOnReadDisableCopyOnRead0read_100_disablecopyonread_adam_m_conv2d_27_bias"/device:CPU:0*
_output_shapes
 �
Read_100/ReadVariableOpReadVariableOp0read_100_disablecopyonread_adam_m_conv2d_27_bias^Read_100/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_200IdentityRead_100/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_201IdentityIdentity_200:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_101/DisableCopyOnReadDisableCopyOnRead0read_101_disablecopyonread_adam_v_conv2d_27_bias"/device:CPU:0*
_output_shapes
 �
Read_101/ReadVariableOpReadVariableOp0read_101_disablecopyonread_adam_v_conv2d_27_bias^Read_101/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_202IdentityRead_101/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_203IdentityIdentity_202:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_102/DisableCopyOnReadDisableCopyOnRead2read_102_disablecopyonread_adam_m_conv2d_28_kernel"/device:CPU:0*
_output_shapes
 �
Read_102/ReadVariableOpReadVariableOp2read_102_disablecopyonread_adam_m_conv2d_28_kernel^Read_102/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0y
Identity_204IdentityRead_102/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@o
Identity_205IdentityIdentity_204:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@�
Read_103/DisableCopyOnReadDisableCopyOnRead2read_103_disablecopyonread_adam_v_conv2d_28_kernel"/device:CPU:0*
_output_shapes
 �
Read_103/ReadVariableOpReadVariableOp2read_103_disablecopyonread_adam_v_conv2d_28_kernel^Read_103/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0y
Identity_206IdentityRead_103/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@o
Identity_207IdentityIdentity_206:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@�
Read_104/DisableCopyOnReadDisableCopyOnRead0read_104_disablecopyonread_adam_m_conv2d_28_bias"/device:CPU:0*
_output_shapes
 �
Read_104/ReadVariableOpReadVariableOp0read_104_disablecopyonread_adam_m_conv2d_28_bias^Read_104/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_208IdentityRead_104/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_209IdentityIdentity_208:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_105/DisableCopyOnReadDisableCopyOnRead0read_105_disablecopyonread_adam_v_conv2d_28_bias"/device:CPU:0*
_output_shapes
 �
Read_105/ReadVariableOpReadVariableOp0read_105_disablecopyonread_adam_v_conv2d_28_bias^Read_105/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_210IdentityRead_105/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_211IdentityIdentity_210:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_106/DisableCopyOnReadDisableCopyOnRead2read_106_disablecopyonread_adam_m_conv2d_29_kernel"/device:CPU:0*
_output_shapes
 �
Read_106/ReadVariableOpReadVariableOp2read_106_disablecopyonread_adam_m_conv2d_29_kernel^Read_106/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0y
Identity_212IdentityRead_106/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@o
Identity_213IdentityIdentity_212:output:0"/device:CPU:0*
T0*&
_output_shapes
:@�
Read_107/DisableCopyOnReadDisableCopyOnRead2read_107_disablecopyonread_adam_v_conv2d_29_kernel"/device:CPU:0*
_output_shapes
 �
Read_107/ReadVariableOpReadVariableOp2read_107_disablecopyonread_adam_v_conv2d_29_kernel^Read_107/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0y
Identity_214IdentityRead_107/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@o
Identity_215IdentityIdentity_214:output:0"/device:CPU:0*
T0*&
_output_shapes
:@�
Read_108/DisableCopyOnReadDisableCopyOnRead0read_108_disablecopyonread_adam_m_conv2d_29_bias"/device:CPU:0*
_output_shapes
 �
Read_108/ReadVariableOpReadVariableOp0read_108_disablecopyonread_adam_m_conv2d_29_bias^Read_108/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_216IdentityRead_108/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_217IdentityIdentity_216:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_109/DisableCopyOnReadDisableCopyOnRead0read_109_disablecopyonread_adam_v_conv2d_29_bias"/device:CPU:0*
_output_shapes
 �
Read_109/ReadVariableOpReadVariableOp0read_109_disablecopyonread_adam_v_conv2d_29_bias^Read_109/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_218IdentityRead_109/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_219IdentityIdentity_218:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_110/DisableCopyOnReadDisableCopyOnRead"read_110_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 �
Read_110/ReadVariableOpReadVariableOp"read_110_disablecopyonread_total_1^Read_110/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i
Identity_220IdentityRead_110/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_221IdentityIdentity_220:output:0"/device:CPU:0*
T0*
_output_shapes
: x
Read_111/DisableCopyOnReadDisableCopyOnRead"read_111_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 �
Read_111/ReadVariableOpReadVariableOp"read_111_disablecopyonread_count_1^Read_111/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i
Identity_222IdentityRead_111/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_223IdentityIdentity_222:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_112/DisableCopyOnReadDisableCopyOnRead read_112_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_112/ReadVariableOpReadVariableOp read_112_disablecopyonread_total^Read_112/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i
Identity_224IdentityRead_112/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_225IdentityIdentity_224:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_113/DisableCopyOnReadDisableCopyOnRead read_113_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_113/ReadVariableOpReadVariableOp read_113_disablecopyonread_count^Read_113/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i
Identity_226IdentityRead_113/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_227IdentityIdentity_226:output:0"/device:CPU:0*
T0*
_output_shapes
: �0
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:s*
dtype0*�/
value�/B�/sB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/55/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/56/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/57/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/58/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/59/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/60/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/61/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/62/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/63/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/64/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/65/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/66/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/67/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/68/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/69/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/70/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/71/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/72/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:s*
dtype0*�
value�B�sB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0Identity_117:output:0Identity_119:output:0Identity_121:output:0Identity_123:output:0Identity_125:output:0Identity_127:output:0Identity_129:output:0Identity_131:output:0Identity_133:output:0Identity_135:output:0Identity_137:output:0Identity_139:output:0Identity_141:output:0Identity_143:output:0Identity_145:output:0Identity_147:output:0Identity_149:output:0Identity_151:output:0Identity_153:output:0Identity_155:output:0Identity_157:output:0Identity_159:output:0Identity_161:output:0Identity_163:output:0Identity_165:output:0Identity_167:output:0Identity_169:output:0Identity_171:output:0Identity_173:output:0Identity_175:output:0Identity_177:output:0Identity_179:output:0Identity_181:output:0Identity_183:output:0Identity_185:output:0Identity_187:output:0Identity_189:output:0Identity_191:output:0Identity_193:output:0Identity_195:output:0Identity_197:output:0Identity_199:output:0Identity_201:output:0Identity_203:output:0Identity_205:output:0Identity_207:output:0Identity_209:output:0Identity_211:output:0Identity_213:output:0Identity_215:output:0Identity_217:output:0Identity_219:output:0Identity_221:output:0Identity_223:output:0Identity_225:output:0Identity_227:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *�
dtypesw
u2s	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_228Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_229IdentityIdentity_228:output:0^NoOp*
T0*
_output_shapes
: �/
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_100/DisableCopyOnRead^Read_100/ReadVariableOp^Read_101/DisableCopyOnRead^Read_101/ReadVariableOp^Read_102/DisableCopyOnRead^Read_102/ReadVariableOp^Read_103/DisableCopyOnRead^Read_103/ReadVariableOp^Read_104/DisableCopyOnRead^Read_104/ReadVariableOp^Read_105/DisableCopyOnRead^Read_105/ReadVariableOp^Read_106/DisableCopyOnRead^Read_106/ReadVariableOp^Read_107/DisableCopyOnRead^Read_107/ReadVariableOp^Read_108/DisableCopyOnRead^Read_108/ReadVariableOp^Read_109/DisableCopyOnRead^Read_109/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_110/DisableCopyOnRead^Read_110/ReadVariableOp^Read_111/DisableCopyOnRead^Read_111/ReadVariableOp^Read_112/DisableCopyOnRead^Read_112/ReadVariableOp^Read_113/DisableCopyOnRead^Read_113/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_58/DisableCopyOnRead^Read_58/ReadVariableOp^Read_59/DisableCopyOnRead^Read_59/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_60/DisableCopyOnRead^Read_60/ReadVariableOp^Read_61/DisableCopyOnRead^Read_61/ReadVariableOp^Read_62/DisableCopyOnRead^Read_62/ReadVariableOp^Read_63/DisableCopyOnRead^Read_63/ReadVariableOp^Read_64/DisableCopyOnRead^Read_64/ReadVariableOp^Read_65/DisableCopyOnRead^Read_65/ReadVariableOp^Read_66/DisableCopyOnRead^Read_66/ReadVariableOp^Read_67/DisableCopyOnRead^Read_67/ReadVariableOp^Read_68/DisableCopyOnRead^Read_68/ReadVariableOp^Read_69/DisableCopyOnRead^Read_69/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_70/DisableCopyOnRead^Read_70/ReadVariableOp^Read_71/DisableCopyOnRead^Read_71/ReadVariableOp^Read_72/DisableCopyOnRead^Read_72/ReadVariableOp^Read_73/DisableCopyOnRead^Read_73/ReadVariableOp^Read_74/DisableCopyOnRead^Read_74/ReadVariableOp^Read_75/DisableCopyOnRead^Read_75/ReadVariableOp^Read_76/DisableCopyOnRead^Read_76/ReadVariableOp^Read_77/DisableCopyOnRead^Read_77/ReadVariableOp^Read_78/DisableCopyOnRead^Read_78/ReadVariableOp^Read_79/DisableCopyOnRead^Read_79/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_80/DisableCopyOnRead^Read_80/ReadVariableOp^Read_81/DisableCopyOnRead^Read_81/ReadVariableOp^Read_82/DisableCopyOnRead^Read_82/ReadVariableOp^Read_83/DisableCopyOnRead^Read_83/ReadVariableOp^Read_84/DisableCopyOnRead^Read_84/ReadVariableOp^Read_85/DisableCopyOnRead^Read_85/ReadVariableOp^Read_86/DisableCopyOnRead^Read_86/ReadVariableOp^Read_87/DisableCopyOnRead^Read_87/ReadVariableOp^Read_88/DisableCopyOnRead^Read_88/ReadVariableOp^Read_89/DisableCopyOnRead^Read_89/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp^Read_90/DisableCopyOnRead^Read_90/ReadVariableOp^Read_91/DisableCopyOnRead^Read_91/ReadVariableOp^Read_92/DisableCopyOnRead^Read_92/ReadVariableOp^Read_93/DisableCopyOnRead^Read_93/ReadVariableOp^Read_94/DisableCopyOnRead^Read_94/ReadVariableOp^Read_95/DisableCopyOnRead^Read_95/ReadVariableOp^Read_96/DisableCopyOnRead^Read_96/ReadVariableOp^Read_97/DisableCopyOnRead^Read_97/ReadVariableOp^Read_98/DisableCopyOnRead^Read_98/ReadVariableOp^Read_99/DisableCopyOnRead^Read_99/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "%
identity_229Identity_229:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp28
Read_100/DisableCopyOnReadRead_100/DisableCopyOnRead22
Read_100/ReadVariableOpRead_100/ReadVariableOp28
Read_101/DisableCopyOnReadRead_101/DisableCopyOnRead22
Read_101/ReadVariableOpRead_101/ReadVariableOp28
Read_102/DisableCopyOnReadRead_102/DisableCopyOnRead22
Read_102/ReadVariableOpRead_102/ReadVariableOp28
Read_103/DisableCopyOnReadRead_103/DisableCopyOnRead22
Read_103/ReadVariableOpRead_103/ReadVariableOp28
Read_104/DisableCopyOnReadRead_104/DisableCopyOnRead22
Read_104/ReadVariableOpRead_104/ReadVariableOp28
Read_105/DisableCopyOnReadRead_105/DisableCopyOnRead22
Read_105/ReadVariableOpRead_105/ReadVariableOp28
Read_106/DisableCopyOnReadRead_106/DisableCopyOnRead22
Read_106/ReadVariableOpRead_106/ReadVariableOp28
Read_107/DisableCopyOnReadRead_107/DisableCopyOnRead22
Read_107/ReadVariableOpRead_107/ReadVariableOp28
Read_108/DisableCopyOnReadRead_108/DisableCopyOnRead22
Read_108/ReadVariableOpRead_108/ReadVariableOp28
Read_109/DisableCopyOnReadRead_109/DisableCopyOnRead22
Read_109/ReadVariableOpRead_109/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp28
Read_110/DisableCopyOnReadRead_110/DisableCopyOnRead22
Read_110/ReadVariableOpRead_110/ReadVariableOp28
Read_111/DisableCopyOnReadRead_111/DisableCopyOnRead22
Read_111/ReadVariableOpRead_111/ReadVariableOp28
Read_112/DisableCopyOnReadRead_112/DisableCopyOnRead22
Read_112/ReadVariableOpRead_112/ReadVariableOp28
Read_113/DisableCopyOnReadRead_113/DisableCopyOnRead22
Read_113/ReadVariableOpRead_113/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp26
Read_57/DisableCopyOnReadRead_57/DisableCopyOnRead20
Read_57/ReadVariableOpRead_57/ReadVariableOp26
Read_58/DisableCopyOnReadRead_58/DisableCopyOnRead20
Read_58/ReadVariableOpRead_58/ReadVariableOp26
Read_59/DisableCopyOnReadRead_59/DisableCopyOnRead20
Read_59/ReadVariableOpRead_59/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp26
Read_60/DisableCopyOnReadRead_60/DisableCopyOnRead20
Read_60/ReadVariableOpRead_60/ReadVariableOp26
Read_61/DisableCopyOnReadRead_61/DisableCopyOnRead20
Read_61/ReadVariableOpRead_61/ReadVariableOp26
Read_62/DisableCopyOnReadRead_62/DisableCopyOnRead20
Read_62/ReadVariableOpRead_62/ReadVariableOp26
Read_63/DisableCopyOnReadRead_63/DisableCopyOnRead20
Read_63/ReadVariableOpRead_63/ReadVariableOp26
Read_64/DisableCopyOnReadRead_64/DisableCopyOnRead20
Read_64/ReadVariableOpRead_64/ReadVariableOp26
Read_65/DisableCopyOnReadRead_65/DisableCopyOnRead20
Read_65/ReadVariableOpRead_65/ReadVariableOp26
Read_66/DisableCopyOnReadRead_66/DisableCopyOnRead20
Read_66/ReadVariableOpRead_66/ReadVariableOp26
Read_67/DisableCopyOnReadRead_67/DisableCopyOnRead20
Read_67/ReadVariableOpRead_67/ReadVariableOp26
Read_68/DisableCopyOnReadRead_68/DisableCopyOnRead20
Read_68/ReadVariableOpRead_68/ReadVariableOp26
Read_69/DisableCopyOnReadRead_69/DisableCopyOnRead20
Read_69/ReadVariableOpRead_69/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp26
Read_70/DisableCopyOnReadRead_70/DisableCopyOnRead20
Read_70/ReadVariableOpRead_70/ReadVariableOp26
Read_71/DisableCopyOnReadRead_71/DisableCopyOnRead20
Read_71/ReadVariableOpRead_71/ReadVariableOp26
Read_72/DisableCopyOnReadRead_72/DisableCopyOnRead20
Read_72/ReadVariableOpRead_72/ReadVariableOp26
Read_73/DisableCopyOnReadRead_73/DisableCopyOnRead20
Read_73/ReadVariableOpRead_73/ReadVariableOp26
Read_74/DisableCopyOnReadRead_74/DisableCopyOnRead20
Read_74/ReadVariableOpRead_74/ReadVariableOp26
Read_75/DisableCopyOnReadRead_75/DisableCopyOnRead20
Read_75/ReadVariableOpRead_75/ReadVariableOp26
Read_76/DisableCopyOnReadRead_76/DisableCopyOnRead20
Read_76/ReadVariableOpRead_76/ReadVariableOp26
Read_77/DisableCopyOnReadRead_77/DisableCopyOnRead20
Read_77/ReadVariableOpRead_77/ReadVariableOp26
Read_78/DisableCopyOnReadRead_78/DisableCopyOnRead20
Read_78/ReadVariableOpRead_78/ReadVariableOp26
Read_79/DisableCopyOnReadRead_79/DisableCopyOnRead20
Read_79/ReadVariableOpRead_79/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp26
Read_80/DisableCopyOnReadRead_80/DisableCopyOnRead20
Read_80/ReadVariableOpRead_80/ReadVariableOp26
Read_81/DisableCopyOnReadRead_81/DisableCopyOnRead20
Read_81/ReadVariableOpRead_81/ReadVariableOp26
Read_82/DisableCopyOnReadRead_82/DisableCopyOnRead20
Read_82/ReadVariableOpRead_82/ReadVariableOp26
Read_83/DisableCopyOnReadRead_83/DisableCopyOnRead20
Read_83/ReadVariableOpRead_83/ReadVariableOp26
Read_84/DisableCopyOnReadRead_84/DisableCopyOnRead20
Read_84/ReadVariableOpRead_84/ReadVariableOp26
Read_85/DisableCopyOnReadRead_85/DisableCopyOnRead20
Read_85/ReadVariableOpRead_85/ReadVariableOp26
Read_86/DisableCopyOnReadRead_86/DisableCopyOnRead20
Read_86/ReadVariableOpRead_86/ReadVariableOp26
Read_87/DisableCopyOnReadRead_87/DisableCopyOnRead20
Read_87/ReadVariableOpRead_87/ReadVariableOp26
Read_88/DisableCopyOnReadRead_88/DisableCopyOnRead20
Read_88/ReadVariableOpRead_88/ReadVariableOp26
Read_89/DisableCopyOnReadRead_89/DisableCopyOnRead20
Read_89/ReadVariableOpRead_89/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp26
Read_90/DisableCopyOnReadRead_90/DisableCopyOnRead20
Read_90/ReadVariableOpRead_90/ReadVariableOp26
Read_91/DisableCopyOnReadRead_91/DisableCopyOnRead20
Read_91/ReadVariableOpRead_91/ReadVariableOp26
Read_92/DisableCopyOnReadRead_92/DisableCopyOnRead20
Read_92/ReadVariableOpRead_92/ReadVariableOp26
Read_93/DisableCopyOnReadRead_93/DisableCopyOnRead20
Read_93/ReadVariableOpRead_93/ReadVariableOp26
Read_94/DisableCopyOnReadRead_94/DisableCopyOnRead20
Read_94/ReadVariableOpRead_94/ReadVariableOp26
Read_95/DisableCopyOnReadRead_95/DisableCopyOnRead20
Read_95/ReadVariableOpRead_95/ReadVariableOp26
Read_96/DisableCopyOnReadRead_96/DisableCopyOnRead20
Read_96/ReadVariableOpRead_96/ReadVariableOp26
Read_97/DisableCopyOnReadRead_97/DisableCopyOnRead20
Read_97/ReadVariableOpRead_97/ReadVariableOp26
Read_98/DisableCopyOnReadRead_98/DisableCopyOnRead20
Read_98/ReadVariableOpRead_98/ReadVariableOp26
Read_99/DisableCopyOnReadRead_99/DisableCopyOnRead20
Read_99/ReadVariableOpRead_99/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:s

_output_shapes
: 
�
�	
(__inference_model_1_layer_call_fn_308270

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@�
	unknown_4:	�%
	unknown_5:��
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�%
	unknown_9:��

unknown_10:	�&

unknown_11:��

unknown_12:	�&

unknown_13:��

unknown_14:	�&

unknown_15:��

unknown_16:	�&

unknown_17:��

unknown_18:	�&

unknown_19:��

unknown_20:	�&

unknown_21:��

unknown_22:	�&

unknown_23:��

unknown_24:	�&

unknown_25:��

unknown_26:	�%

unknown_27:@�

unknown_28:@%

unknown_29:�@

unknown_30:@$

unknown_31:@@

unknown_32:@$

unknown_33:@

unknown_34:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*F
_read_only_resource_inputs(
&$	
 !"#$*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_307535w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:���������@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
�
E__inference_conv2d_17_layer_call_and_return_conditional_losses_308767

inputs9
conv2d_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������  �j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������  �w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������  @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������  @
 
_user_specified_nameinputs
�
Y
#__inference__update_step_xla_297865
gradient$
variable:��*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*)
_input_shapes
:��: *
	_noinline(:R N
(
_output_shapes
:��
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
s
I__inference_concatenate_4_layer_call_and_return_conditional_losses_307230

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :~
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:���������  �`
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:���������  �"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������  �:���������  �:X T
0
_output_shapes
:���������  �
 
_user_specified_nameinputs:XT
0
_output_shapes
:���������  �
 
_user_specified_nameinputs
�
Y
#__inference__update_step_xla_297905
gradient$
variable:��*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*)
_input_shapes
:��: *
	_noinline(:R N
(
_output_shapes
:��
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
Z
.__inference_concatenate_5_layer_call_fn_309125
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_concatenate_5_layer_call_and_return_conditional_losses_307278i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:���������@@�"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������@@@:���������@@@:Y U
/
_output_shapes
:���������@@@
"
_user_specified_name
inputs_0:YU
/
_output_shapes
:���������@@@
"
_user_specified_name
inputs_1
�
�
*__inference_conv2d_17_layer_call_fn_308756

inputs"
unknown:@�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_17_layer_call_and_return_conditional_losses_307077x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������  �`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������  @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  @
 
_user_specified_nameinputs
�
W
#__inference__update_step_xla_297815
gradient"
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:@: *
	_noinline(:P L
&
_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
�
*__inference_conv2d_18_layer_call_fn_308776

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_18_layer_call_and_return_conditional_losses_307094x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������  �`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������  �: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������  �
 
_user_specified_nameinputs
�
�
*__inference_conv2d_16_layer_call_fn_308726

inputs!
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_16_layer_call_and_return_conditional_losses_307059w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@@@
 
_user_specified_nameinputs
�
X
#__inference__update_step_xla_297835
gradient#
variable:@�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*(
_input_shapes
:@�: *
	_noinline(:Q M
'
_output_shapes
:@�
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
�
E__inference_conv2d_23_layer_call_and_return_conditional_losses_308962

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
K
#__inference__update_step_xla_297970
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
K
#__inference__update_step_xla_297980
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
Y
#__inference__update_step_xla_297945
gradient$
variable:��*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*)
_input_shapes
:��: *
	_noinline(:R N
(
_output_shapes
:��
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
X
#__inference__update_step_xla_297965
gradient#
variable:�@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*(
_input_shapes
:�@: *
	_noinline(:Q M
'
_output_shapes
:�@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
��
�L
"__inference__traced_restore_310251
file_prefix;
!assignvariableop_conv2d_15_kernel:@/
!assignvariableop_1_conv2d_15_bias:@=
#assignvariableop_2_conv2d_16_kernel:@@/
!assignvariableop_3_conv2d_16_bias:@>
#assignvariableop_4_conv2d_17_kernel:@�0
!assignvariableop_5_conv2d_17_bias:	�?
#assignvariableop_6_conv2d_18_kernel:��0
!assignvariableop_7_conv2d_18_bias:	�?
#assignvariableop_8_conv2d_19_kernel:��0
!assignvariableop_9_conv2d_19_bias:	�@
$assignvariableop_10_conv2d_20_kernel:��1
"assignvariableop_11_conv2d_20_bias:	�@
$assignvariableop_12_conv2d_21_kernel:��1
"assignvariableop_13_conv2d_21_bias:	�@
$assignvariableop_14_conv2d_22_kernel:��1
"assignvariableop_15_conv2d_22_bias:	�I
-assignvariableop_16_conv2d_transpose_3_kernel:��:
+assignvariableop_17_conv2d_transpose_3_bias:	�@
$assignvariableop_18_conv2d_23_kernel:��1
"assignvariableop_19_conv2d_23_bias:	�@
$assignvariableop_20_conv2d_24_kernel:��1
"assignvariableop_21_conv2d_24_bias:	�I
-assignvariableop_22_conv2d_transpose_4_kernel:��:
+assignvariableop_23_conv2d_transpose_4_bias:	�@
$assignvariableop_24_conv2d_25_kernel:��1
"assignvariableop_25_conv2d_25_bias:	�@
$assignvariableop_26_conv2d_26_kernel:��1
"assignvariableop_27_conv2d_26_bias:	�H
-assignvariableop_28_conv2d_transpose_5_kernel:@�9
+assignvariableop_29_conv2d_transpose_5_bias:@?
$assignvariableop_30_conv2d_27_kernel:�@0
"assignvariableop_31_conv2d_27_bias:@>
$assignvariableop_32_conv2d_28_kernel:@@0
"assignvariableop_33_conv2d_28_bias:@>
$assignvariableop_34_conv2d_29_kernel:@0
"assignvariableop_35_conv2d_29_bias:'
assignvariableop_36_iteration:	 +
!assignvariableop_37_learning_rate: E
+assignvariableop_38_adam_m_conv2d_15_kernel:@E
+assignvariableop_39_adam_v_conv2d_15_kernel:@7
)assignvariableop_40_adam_m_conv2d_15_bias:@7
)assignvariableop_41_adam_v_conv2d_15_bias:@E
+assignvariableop_42_adam_m_conv2d_16_kernel:@@E
+assignvariableop_43_adam_v_conv2d_16_kernel:@@7
)assignvariableop_44_adam_m_conv2d_16_bias:@7
)assignvariableop_45_adam_v_conv2d_16_bias:@F
+assignvariableop_46_adam_m_conv2d_17_kernel:@�F
+assignvariableop_47_adam_v_conv2d_17_kernel:@�8
)assignvariableop_48_adam_m_conv2d_17_bias:	�8
)assignvariableop_49_adam_v_conv2d_17_bias:	�G
+assignvariableop_50_adam_m_conv2d_18_kernel:��G
+assignvariableop_51_adam_v_conv2d_18_kernel:��8
)assignvariableop_52_adam_m_conv2d_18_bias:	�8
)assignvariableop_53_adam_v_conv2d_18_bias:	�G
+assignvariableop_54_adam_m_conv2d_19_kernel:��G
+assignvariableop_55_adam_v_conv2d_19_kernel:��8
)assignvariableop_56_adam_m_conv2d_19_bias:	�8
)assignvariableop_57_adam_v_conv2d_19_bias:	�G
+assignvariableop_58_adam_m_conv2d_20_kernel:��G
+assignvariableop_59_adam_v_conv2d_20_kernel:��8
)assignvariableop_60_adam_m_conv2d_20_bias:	�8
)assignvariableop_61_adam_v_conv2d_20_bias:	�G
+assignvariableop_62_adam_m_conv2d_21_kernel:��G
+assignvariableop_63_adam_v_conv2d_21_kernel:��8
)assignvariableop_64_adam_m_conv2d_21_bias:	�8
)assignvariableop_65_adam_v_conv2d_21_bias:	�G
+assignvariableop_66_adam_m_conv2d_22_kernel:��G
+assignvariableop_67_adam_v_conv2d_22_kernel:��8
)assignvariableop_68_adam_m_conv2d_22_bias:	�8
)assignvariableop_69_adam_v_conv2d_22_bias:	�P
4assignvariableop_70_adam_m_conv2d_transpose_3_kernel:��P
4assignvariableop_71_adam_v_conv2d_transpose_3_kernel:��A
2assignvariableop_72_adam_m_conv2d_transpose_3_bias:	�A
2assignvariableop_73_adam_v_conv2d_transpose_3_bias:	�G
+assignvariableop_74_adam_m_conv2d_23_kernel:��G
+assignvariableop_75_adam_v_conv2d_23_kernel:��8
)assignvariableop_76_adam_m_conv2d_23_bias:	�8
)assignvariableop_77_adam_v_conv2d_23_bias:	�G
+assignvariableop_78_adam_m_conv2d_24_kernel:��G
+assignvariableop_79_adam_v_conv2d_24_kernel:��8
)assignvariableop_80_adam_m_conv2d_24_bias:	�8
)assignvariableop_81_adam_v_conv2d_24_bias:	�P
4assignvariableop_82_adam_m_conv2d_transpose_4_kernel:��P
4assignvariableop_83_adam_v_conv2d_transpose_4_kernel:��A
2assignvariableop_84_adam_m_conv2d_transpose_4_bias:	�A
2assignvariableop_85_adam_v_conv2d_transpose_4_bias:	�G
+assignvariableop_86_adam_m_conv2d_25_kernel:��G
+assignvariableop_87_adam_v_conv2d_25_kernel:��8
)assignvariableop_88_adam_m_conv2d_25_bias:	�8
)assignvariableop_89_adam_v_conv2d_25_bias:	�G
+assignvariableop_90_adam_m_conv2d_26_kernel:��G
+assignvariableop_91_adam_v_conv2d_26_kernel:��8
)assignvariableop_92_adam_m_conv2d_26_bias:	�8
)assignvariableop_93_adam_v_conv2d_26_bias:	�O
4assignvariableop_94_adam_m_conv2d_transpose_5_kernel:@�O
4assignvariableop_95_adam_v_conv2d_transpose_5_kernel:@�@
2assignvariableop_96_adam_m_conv2d_transpose_5_bias:@@
2assignvariableop_97_adam_v_conv2d_transpose_5_bias:@F
+assignvariableop_98_adam_m_conv2d_27_kernel:�@F
+assignvariableop_99_adam_v_conv2d_27_kernel:�@8
*assignvariableop_100_adam_m_conv2d_27_bias:@8
*assignvariableop_101_adam_v_conv2d_27_bias:@F
,assignvariableop_102_adam_m_conv2d_28_kernel:@@F
,assignvariableop_103_adam_v_conv2d_28_kernel:@@8
*assignvariableop_104_adam_m_conv2d_28_bias:@8
*assignvariableop_105_adam_v_conv2d_28_bias:@F
,assignvariableop_106_adam_m_conv2d_29_kernel:@F
,assignvariableop_107_adam_v_conv2d_29_kernel:@8
*assignvariableop_108_adam_m_conv2d_29_bias:8
*assignvariableop_109_adam_v_conv2d_29_bias:&
assignvariableop_110_total_1: &
assignvariableop_111_count_1: $
assignvariableop_112_total: $
assignvariableop_113_count: 
identity_115��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_100�AssignVariableOp_101�AssignVariableOp_102�AssignVariableOp_103�AssignVariableOp_104�AssignVariableOp_105�AssignVariableOp_106�AssignVariableOp_107�AssignVariableOp_108�AssignVariableOp_109�AssignVariableOp_11�AssignVariableOp_110�AssignVariableOp_111�AssignVariableOp_112�AssignVariableOp_113�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_79�AssignVariableOp_8�AssignVariableOp_80�AssignVariableOp_81�AssignVariableOp_82�AssignVariableOp_83�AssignVariableOp_84�AssignVariableOp_85�AssignVariableOp_86�AssignVariableOp_87�AssignVariableOp_88�AssignVariableOp_89�AssignVariableOp_9�AssignVariableOp_90�AssignVariableOp_91�AssignVariableOp_92�AssignVariableOp_93�AssignVariableOp_94�AssignVariableOp_95�AssignVariableOp_96�AssignVariableOp_97�AssignVariableOp_98�AssignVariableOp_99�0
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:s*
dtype0*�/
value�/B�/sB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/55/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/56/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/57/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/58/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/59/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/60/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/61/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/62/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/63/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/64/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/65/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/66/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/67/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/68/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/69/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/70/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/71/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/72/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:s*
dtype0*�
value�B�sB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*�
dtypesw
u2s	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_15_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_15_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_16_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_16_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_17_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_17_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_18_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_18_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv2d_19_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv2d_19_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp$assignvariableop_10_conv2d_20_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp"assignvariableop_11_conv2d_20_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv2d_21_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv2d_21_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp$assignvariableop_14_conv2d_22_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp"assignvariableop_15_conv2d_22_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp-assignvariableop_16_conv2d_transpose_3_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp+assignvariableop_17_conv2d_transpose_3_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp$assignvariableop_18_conv2d_23_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp"assignvariableop_19_conv2d_23_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp$assignvariableop_20_conv2d_24_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp"assignvariableop_21_conv2d_24_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp-assignvariableop_22_conv2d_transpose_4_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp+assignvariableop_23_conv2d_transpose_4_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp$assignvariableop_24_conv2d_25_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp"assignvariableop_25_conv2d_25_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp$assignvariableop_26_conv2d_26_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp"assignvariableop_27_conv2d_26_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp-assignvariableop_28_conv2d_transpose_5_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp+assignvariableop_29_conv2d_transpose_5_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp$assignvariableop_30_conv2d_27_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp"assignvariableop_31_conv2d_27_biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp$assignvariableop_32_conv2d_28_kernelIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp"assignvariableop_33_conv2d_28_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp$assignvariableop_34_conv2d_29_kernelIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp"assignvariableop_35_conv2d_29_biasIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_36AssignVariableOpassignvariableop_36_iterationIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp!assignvariableop_37_learning_rateIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp+assignvariableop_38_adam_m_conv2d_15_kernelIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_v_conv2d_15_kernelIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_m_conv2d_15_biasIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp)assignvariableop_41_adam_v_conv2d_15_biasIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp+assignvariableop_42_adam_m_conv2d_16_kernelIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_v_conv2d_16_kernelIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_m_conv2d_16_biasIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp)assignvariableop_45_adam_v_conv2d_16_biasIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp+assignvariableop_46_adam_m_conv2d_17_kernelIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_v_conv2d_17_kernelIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_m_conv2d_17_biasIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp)assignvariableop_49_adam_v_conv2d_17_biasIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp+assignvariableop_50_adam_m_conv2d_18_kernelIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp+assignvariableop_51_adam_v_conv2d_18_kernelIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp)assignvariableop_52_adam_m_conv2d_18_biasIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp)assignvariableop_53_adam_v_conv2d_18_biasIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp+assignvariableop_54_adam_m_conv2d_19_kernelIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp+assignvariableop_55_adam_v_conv2d_19_kernelIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp)assignvariableop_56_adam_m_conv2d_19_biasIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp)assignvariableop_57_adam_v_conv2d_19_biasIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp+assignvariableop_58_adam_m_conv2d_20_kernelIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp+assignvariableop_59_adam_v_conv2d_20_kernelIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOp)assignvariableop_60_adam_m_conv2d_20_biasIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOp)assignvariableop_61_adam_v_conv2d_20_biasIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOp+assignvariableop_62_adam_m_conv2d_21_kernelIdentity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOp+assignvariableop_63_adam_v_conv2d_21_kernelIdentity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOp)assignvariableop_64_adam_m_conv2d_21_biasIdentity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOp)assignvariableop_65_adam_v_conv2d_21_biasIdentity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOp+assignvariableop_66_adam_m_conv2d_22_kernelIdentity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOp+assignvariableop_67_adam_v_conv2d_22_kernelIdentity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOp)assignvariableop_68_adam_m_conv2d_22_biasIdentity_68:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOp)assignvariableop_69_adam_v_conv2d_22_biasIdentity_69:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_70AssignVariableOp4assignvariableop_70_adam_m_conv2d_transpose_3_kernelIdentity_70:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_71AssignVariableOp4assignvariableop_71_adam_v_conv2d_transpose_3_kernelIdentity_71:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_72AssignVariableOp2assignvariableop_72_adam_m_conv2d_transpose_3_biasIdentity_72:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_73AssignVariableOp2assignvariableop_73_adam_v_conv2d_transpose_3_biasIdentity_73:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_74AssignVariableOp+assignvariableop_74_adam_m_conv2d_23_kernelIdentity_74:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_75AssignVariableOp+assignvariableop_75_adam_v_conv2d_23_kernelIdentity_75:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_76AssignVariableOp)assignvariableop_76_adam_m_conv2d_23_biasIdentity_76:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_77AssignVariableOp)assignvariableop_77_adam_v_conv2d_23_biasIdentity_77:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_78AssignVariableOp+assignvariableop_78_adam_m_conv2d_24_kernelIdentity_78:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_79AssignVariableOp+assignvariableop_79_adam_v_conv2d_24_kernelIdentity_79:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_80AssignVariableOp)assignvariableop_80_adam_m_conv2d_24_biasIdentity_80:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_81AssignVariableOp)assignvariableop_81_adam_v_conv2d_24_biasIdentity_81:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_82AssignVariableOp4assignvariableop_82_adam_m_conv2d_transpose_4_kernelIdentity_82:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_83AssignVariableOp4assignvariableop_83_adam_v_conv2d_transpose_4_kernelIdentity_83:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_84AssignVariableOp2assignvariableop_84_adam_m_conv2d_transpose_4_biasIdentity_84:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_85AssignVariableOp2assignvariableop_85_adam_v_conv2d_transpose_4_biasIdentity_85:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_86AssignVariableOp+assignvariableop_86_adam_m_conv2d_25_kernelIdentity_86:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_87AssignVariableOp+assignvariableop_87_adam_v_conv2d_25_kernelIdentity_87:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_88AssignVariableOp)assignvariableop_88_adam_m_conv2d_25_biasIdentity_88:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_89AssignVariableOp)assignvariableop_89_adam_v_conv2d_25_biasIdentity_89:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_90AssignVariableOp+assignvariableop_90_adam_m_conv2d_26_kernelIdentity_90:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_91AssignVariableOp+assignvariableop_91_adam_v_conv2d_26_kernelIdentity_91:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_92AssignVariableOp)assignvariableop_92_adam_m_conv2d_26_biasIdentity_92:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_93AssignVariableOp)assignvariableop_93_adam_v_conv2d_26_biasIdentity_93:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_94AssignVariableOp4assignvariableop_94_adam_m_conv2d_transpose_5_kernelIdentity_94:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_95AssignVariableOp4assignvariableop_95_adam_v_conv2d_transpose_5_kernelIdentity_95:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_96AssignVariableOp2assignvariableop_96_adam_m_conv2d_transpose_5_biasIdentity_96:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_97AssignVariableOp2assignvariableop_97_adam_v_conv2d_transpose_5_biasIdentity_97:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_98AssignVariableOp+assignvariableop_98_adam_m_conv2d_27_kernelIdentity_98:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_99AssignVariableOp+assignvariableop_99_adam_v_conv2d_27_kernelIdentity_99:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_100AssignVariableOp*assignvariableop_100_adam_m_conv2d_27_biasIdentity_100:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_101AssignVariableOp*assignvariableop_101_adam_v_conv2d_27_biasIdentity_101:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_102AssignVariableOp,assignvariableop_102_adam_m_conv2d_28_kernelIdentity_102:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_103AssignVariableOp,assignvariableop_103_adam_v_conv2d_28_kernelIdentity_103:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_104AssignVariableOp*assignvariableop_104_adam_m_conv2d_28_biasIdentity_104:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_105AssignVariableOp*assignvariableop_105_adam_v_conv2d_28_biasIdentity_105:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_106AssignVariableOp,assignvariableop_106_adam_m_conv2d_29_kernelIdentity_106:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_107AssignVariableOp,assignvariableop_107_adam_v_conv2d_29_kernelIdentity_107:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_108AssignVariableOp*assignvariableop_108_adam_m_conv2d_29_biasIdentity_108:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_109AssignVariableOp*assignvariableop_109_adam_v_conv2d_29_biasIdentity_109:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_110AssignVariableOpassignvariableop_110_total_1Identity_110:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_111AssignVariableOpassignvariableop_111_count_1Identity_111:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_112AssignVariableOpassignvariableop_112_totalIdentity_112:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_113AssignVariableOpassignvariableop_113_countIdentity_113:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_114Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_115IdentityIdentity_114:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 "%
identity_115Identity_115:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
L
#__inference__update_step_xla_297930
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:�: *
	_noinline(:E A

_output_shapes	
:�
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
L
#__inference__update_step_xla_297890
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:�: *
	_noinline(:E A

_output_shapes	
:�
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
L
#__inference__update_step_xla_297860
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:�: *
	_noinline(:E A

_output_shapes	
:�
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
�
E__inference_conv2d_26_layer_call_and_return_conditional_losses_309077

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������  �j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������  �w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������  �: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������  �
 
_user_specified_nameinputs
�
�
E__inference_conv2d_23_layer_call_and_return_conditional_losses_307195

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
g
K__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_308847

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
s
I__inference_concatenate_3_layer_call_and_return_conditional_losses_307182

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :~
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:����������`
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:����������:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs:XT
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
E__inference_conv2d_16_layer_call_and_return_conditional_losses_308737

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@@@
 
_user_specified_nameinputs
�
u
I__inference_concatenate_3_layer_call_and_return_conditional_losses_308942
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:����������`
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:����������:����������:Z V
0
_output_shapes
:����������
"
_user_specified_name
inputs_0:ZV
0
_output_shapes
:����������
"
_user_specified_name
inputs_1
�
�
*__inference_conv2d_28_layer_call_fn_309161

inputs!
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_28_layer_call_and_return_conditional_losses_307308w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@@@
 
_user_specified_nameinputs
�
Z
.__inference_concatenate_3_layer_call_fn_308935
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_concatenate_3_layer_call_and_return_conditional_losses_307182i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:����������:����������:Z V
0
_output_shapes
:����������
"
_user_specified_name
inputs_0:ZV
0
_output_shapes
:����������
"
_user_specified_name
inputs_1
�
W
#__inference__update_step_xla_297975
gradient"
variable:@@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:@@: *
	_noinline(:P L
&
_output_shapes
:@@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
�
*__inference_conv2d_27_layer_call_fn_309141

inputs"
unknown:�@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_27_layer_call_and_return_conditional_losses_307291w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
�
E__inference_conv2d_24_layer_call_and_return_conditional_losses_307212

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
g
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_308797

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
L
#__inference__update_step_xla_297880
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:�: *
	_noinline(:E A

_output_shapes	
:�
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
�
E__inference_conv2d_21_layer_call_and_return_conditional_losses_308867

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
Y
#__inference__update_step_xla_297855
gradient$
variable:��*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*)
_input_shapes
:��: *
	_noinline(:R N
(
_output_shapes
:��
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
�
3__inference_conv2d_transpose_5_layer_call_fn_309086

inputs"
unknown:@�
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_307017�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
E__inference_conv2d_25_layer_call_and_return_conditional_losses_309057

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������  �j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������  �w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������  �: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������  �
 
_user_specified_nameinputs
�
Y
#__inference__update_step_xla_297845
gradient$
variable:��*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*)
_input_shapes
:��: *
	_noinline(:R N
(
_output_shapes
:��
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
�
E__inference_conv2d_27_layer_call_and_return_conditional_losses_309152

inputs9
conv2d_readvariableop_resource:�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:�@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������@@�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������@@�
 
_user_specified_nameinputs
�
Y
#__inference__update_step_xla_297925
gradient$
variable:��*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*)
_input_shapes
:��: *
	_noinline(:R N
(
_output_shapes
:��
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
�
E__inference_conv2d_22_layer_call_and_return_conditional_losses_307164

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
*__inference_conv2d_24_layer_call_fn_308971

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_24_layer_call_and_return_conditional_losses_307212x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
g
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_306877

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
3__inference_conv2d_transpose_4_layer_call_fn_308991

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_306973�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
g
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_306865

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
*__inference_conv2d_20_layer_call_fn_308826

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_20_layer_call_and_return_conditional_losses_307129x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
L
0__inference_max_pooling2d_4_layer_call_fn_308792

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_306877�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
E__inference_conv2d_15_layer_call_and_return_conditional_losses_308717

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
L
#__inference__update_step_xla_297940
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:�: *
	_noinline(:E A

_output_shapes	
:�
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
�
E__inference_conv2d_24_layer_call_and_return_conditional_losses_308982

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
E__inference_conv2d_19_layer_call_and_return_conditional_losses_307112

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�s
�
C__inference_model_1_layer_call_and_return_conditional_losses_307535

inputs*
conv2d_15_307438:@
conv2d_15_307440:@*
conv2d_16_307443:@@
conv2d_16_307445:@+
conv2d_17_307449:@�
conv2d_17_307451:	�,
conv2d_18_307454:��
conv2d_18_307456:	�,
conv2d_19_307460:��
conv2d_19_307462:	�,
conv2d_20_307465:��
conv2d_20_307467:	�,
conv2d_21_307471:��
conv2d_21_307473:	�,
conv2d_22_307476:��
conv2d_22_307478:	�5
conv2d_transpose_3_307481:��(
conv2d_transpose_3_307483:	�,
conv2d_23_307487:��
conv2d_23_307489:	�,
conv2d_24_307492:��
conv2d_24_307494:	�5
conv2d_transpose_4_307497:��(
conv2d_transpose_4_307499:	�,
conv2d_25_307503:��
conv2d_25_307505:	�,
conv2d_26_307508:��
conv2d_26_307510:	�4
conv2d_transpose_5_307513:@�'
conv2d_transpose_5_307515:@+
conv2d_27_307519:�@
conv2d_27_307521:@*
conv2d_28_307524:@@
conv2d_28_307526:@*
conv2d_29_307529:@
conv2d_29_307531:
identity��!conv2d_15/StatefulPartitionedCall�!conv2d_16/StatefulPartitionedCall�!conv2d_17/StatefulPartitionedCall�!conv2d_18/StatefulPartitionedCall�!conv2d_19/StatefulPartitionedCall�!conv2d_20/StatefulPartitionedCall�!conv2d_21/StatefulPartitionedCall�!conv2d_22/StatefulPartitionedCall�!conv2d_23/StatefulPartitionedCall�!conv2d_24/StatefulPartitionedCall�!conv2d_25/StatefulPartitionedCall�!conv2d_26/StatefulPartitionedCall�!conv2d_27/StatefulPartitionedCall�!conv2d_28/StatefulPartitionedCall�!conv2d_29/StatefulPartitionedCall�*conv2d_transpose_3/StatefulPartitionedCall�*conv2d_transpose_4/StatefulPartitionedCall�*conv2d_transpose_5/StatefulPartitionedCall�
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_15_307438conv2d_15_307440*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_307042�
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0conv2d_16_307443conv2d_16_307445*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_16_layer_call_and_return_conditional_losses_307059�
max_pooling2d_3/PartitionedCallPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_306865�
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv2d_17_307449conv2d_17_307451*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_17_layer_call_and_return_conditional_losses_307077�
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0conv2d_18_307454conv2d_18_307456*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_18_layer_call_and_return_conditional_losses_307094�
max_pooling2d_4/PartitionedCallPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_306877�
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_19_307460conv2d_19_307462*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_19_layer_call_and_return_conditional_losses_307112�
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0conv2d_20_307465conv2d_20_307467*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_20_layer_call_and_return_conditional_losses_307129�
max_pooling2d_5/PartitionedCallPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_306889�
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0conv2d_21_307471conv2d_21_307473*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_21_layer_call_and_return_conditional_losses_307147�
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0conv2d_22_307476conv2d_22_307478*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_22_layer_call_and_return_conditional_losses_307164�
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0conv2d_transpose_3_307481conv2d_transpose_3_307483*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_306929�
concatenate_3/PartitionedCallPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0*conv2d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_concatenate_3_layer_call_and_return_conditional_losses_307182�
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0conv2d_23_307487conv2d_23_307489*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_23_layer_call_and_return_conditional_losses_307195�
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0conv2d_24_307492conv2d_24_307494*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_24_layer_call_and_return_conditional_losses_307212�
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0conv2d_transpose_4_307497conv2d_transpose_4_307499*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_306973�
concatenate_4/PartitionedCallPartitionedCall3conv2d_transpose_4/StatefulPartitionedCall:output:0*conv2d_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_concatenate_4_layer_call_and_return_conditional_losses_307230�
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0conv2d_25_307503conv2d_25_307505*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_25_layer_call_and_return_conditional_losses_307243�
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0conv2d_26_307508conv2d_26_307510*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_26_layer_call_and_return_conditional_losses_307260�
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0conv2d_transpose_5_307513conv2d_transpose_5_307515*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_307017�
concatenate_5/PartitionedCallPartitionedCall3conv2d_transpose_5/StatefulPartitionedCall:output:0*conv2d_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������@@�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_concatenate_5_layer_call_and_return_conditional_losses_307278�
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0conv2d_27_307519conv2d_27_307521*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_27_layer_call_and_return_conditional_losses_307291�
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0conv2d_28_307524conv2d_28_307526*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_28_layer_call_and_return_conditional_losses_307308�
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0conv2d_29_307529conv2d_29_307531*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_29_layer_call_and_return_conditional_losses_307325�
IdentityIdentity*conv2d_29/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@�
NoOpNoOp"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall+^conv2d_transpose_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:���������@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
�	
(__inference_model_1_layer_call_fn_308347

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@�
	unknown_4:	�%
	unknown_5:��
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�%
	unknown_9:��

unknown_10:	�&

unknown_11:��

unknown_12:	�&

unknown_13:��

unknown_14:	�&

unknown_15:��

unknown_16:	�&

unknown_17:��

unknown_18:	�&

unknown_19:��

unknown_20:	�&

unknown_21:��

unknown_22:	�&

unknown_23:��

unknown_24:	�&

unknown_25:��

unknown_26:	�%

unknown_27:@�

unknown_28:@%

unknown_29:�@

unknown_30:@$

unknown_31:@@

unknown_32:@$

unknown_33:@

unknown_34:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*F
_read_only_resource_inputs(
&$	
 !"#$*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_307712w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:���������@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
Y
#__inference__update_step_xla_297895
gradient$
variable:��*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*)
_input_shapes
:��: *
	_noinline(:R N
(
_output_shapes
:��
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
�
E__inference_conv2d_18_layer_call_and_return_conditional_losses_308787

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������  �j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������  �w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������  �: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������  �
 
_user_specified_nameinputs
�
K
#__inference__update_step_xla_297960
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
� 
�
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_309024

inputsD
(conv2d_transpose_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :�y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������z
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
� 
�
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_306929

inputsD
(conv2d_transpose_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :�y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������z
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
E__inference_conv2d_17_layer_call_and_return_conditional_losses_307077

inputs9
conv2d_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������  �j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������  �w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������  @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������  @
 
_user_specified_nameinputs
�
�
E__inference_conv2d_26_layer_call_and_return_conditional_losses_307260

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������  �j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������  �w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������  �: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������  �
 
_user_specified_nameinputs
�
K
#__inference__update_step_xla_297830
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
L
#__inference__update_step_xla_297950
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:�: *
	_noinline(:E A

_output_shapes	
:�
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
�	
$__inference_signature_wrapper_308193
input_2!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@�
	unknown_4:	�%
	unknown_5:��
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�%
	unknown_9:��

unknown_10:	�&

unknown_11:��

unknown_12:	�&

unknown_13:��

unknown_14:	�&

unknown_15:��

unknown_16:	�&

unknown_17:��

unknown_18:	�&

unknown_19:��

unknown_20:	�&

unknown_21:��

unknown_22:	�&

unknown_23:��

unknown_24:	�&

unknown_25:��

unknown_26:	�%

unknown_27:@�

unknown_28:@%

unknown_29:�@

unknown_30:@$

unknown_31:@@

unknown_32:@$

unknown_33:@

unknown_34:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@*F
_read_only_resource_inputs(
&$	
 !"#$*0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__wrapped_model_306859w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:���������@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������@@
!
_user_specified_name	input_2
� 
�
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_308929

inputsD
(conv2d_transpose_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :�y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������z
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
input_28
serving_default_input_2:0���������@@E
	conv2d_298
StatefulPartitionedCall:0���������@@tensorflow/serving/predict:�
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
layer_with_weights-8
layer-12
layer-13
layer_with_weights-9
layer-14
layer_with_weights-10
layer-15
layer_with_weights-11
layer-16
layer-17
layer_with_weights-12
layer-18
layer_with_weights-13
layer-19
layer_with_weights-14
layer-20
layer-21
layer_with_weights-15
layer-22
layer_with_weights-16
layer-23
layer_with_weights-17
layer-24
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
 _default_save_signature
!	optimizer
"
signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses

)kernel
*bias
 +_jit_compiled_convolution_op"
_tf_keras_layer
�
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses

2kernel
3bias
 4_jit_compiled_convolution_op"
_tf_keras_layer
�
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses"
_tf_keras_layer
�
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses

Akernel
Bbias
 C_jit_compiled_convolution_op"
_tf_keras_layer
�
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses

Jkernel
Kbias
 L_jit_compiled_convolution_op"
_tf_keras_layer
�
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses"
_tf_keras_layer
�
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses

Ykernel
Zbias
 [_jit_compiled_convolution_op"
_tf_keras_layer
�
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses

bkernel
cbias
 d_jit_compiled_convolution_op"
_tf_keras_layer
�
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses"
_tf_keras_layer
�
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses

qkernel
rbias
 s_jit_compiled_convolution_op"
_tf_keras_layer
�
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses

zkernel
{bias
 |_jit_compiled_convolution_op"
_tf_keras_layer
�
}	variables
~trainable_variables
regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
)0
*1
22
33
A4
B5
J6
K7
Y8
Z9
b10
c11
q12
r13
z14
{15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35"
trackable_list_wrapper
�
)0
*1
22
33
A4
B5
J6
K7
Y8
Z9
b10
c11
q12
r13
z14
{15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
 _default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
(__inference_model_1_layer_call_fn_307610
(__inference_model_1_layer_call_fn_307787
(__inference_model_1_layer_call_fn_308270
(__inference_model_1_layer_call_fn_308347�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
C__inference_model_1_layer_call_and_return_conditional_losses_307332
C__inference_model_1_layer_call_and_return_conditional_losses_307432
C__inference_model_1_layer_call_and_return_conditional_losses_308522
C__inference_model_1_layer_call_and_return_conditional_losses_308697�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�B�
!__inference__wrapped_model_306859input_2"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla"
experimentalOptimizer
-
�serving_default"
signature_map
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_15_layer_call_fn_308706�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_15_layer_call_and_return_conditional_losses_308717�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
*:(@2conv2d_15/kernel
:@2conv2d_15/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_16_layer_call_fn_308726�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_16_layer_call_and_return_conditional_losses_308737�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
*:(@@2conv2d_16/kernel
:@2conv2d_16/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_max_pooling2d_3_layer_call_fn_308742�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_308747�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_17_layer_call_fn_308756�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_17_layer_call_and_return_conditional_losses_308767�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
+:)@�2conv2d_17/kernel
:�2conv2d_17/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
J0
K1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_18_layer_call_fn_308776�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_18_layer_call_and_return_conditional_losses_308787�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
,:*��2conv2d_18/kernel
:�2conv2d_18/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_max_pooling2d_4_layer_call_fn_308792�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_308797�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_19_layer_call_fn_308806�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_19_layer_call_and_return_conditional_losses_308817�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
,:*��2conv2d_19/kernel
:�2conv2d_19/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
b0
c1"
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_20_layer_call_fn_308826�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_20_layer_call_and_return_conditional_losses_308837�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
,:*��2conv2d_20/kernel
:�2conv2d_20/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_max_pooling2d_5_layer_call_fn_308842�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_308847�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
q0
r1"
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_21_layer_call_fn_308856�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_21_layer_call_and_return_conditional_losses_308867�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
,:*��2conv2d_21/kernel
:�2conv2d_21/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
z0
{1"
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_22_layer_call_fn_308876�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_22_layer_call_and_return_conditional_losses_308887�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
,:*��2conv2d_22/kernel
:�2conv2d_22/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
}	variables
~trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
3__inference_conv2d_transpose_3_layer_call_fn_308896�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_308929�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
5:3��2conv2d_transpose_3/kernel
&:$�2conv2d_transpose_3/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_concatenate_3_layer_call_fn_308935�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_concatenate_3_layer_call_and_return_conditional_losses_308942�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_23_layer_call_fn_308951�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_23_layer_call_and_return_conditional_losses_308962�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
,:*��2conv2d_23/kernel
:�2conv2d_23/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_24_layer_call_fn_308971�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_24_layer_call_and_return_conditional_losses_308982�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
,:*��2conv2d_24/kernel
:�2conv2d_24/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
3__inference_conv2d_transpose_4_layer_call_fn_308991�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_309024�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
5:3��2conv2d_transpose_4/kernel
&:$�2conv2d_transpose_4/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_concatenate_4_layer_call_fn_309030�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_concatenate_4_layer_call_and_return_conditional_losses_309037�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_25_layer_call_fn_309046�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_25_layer_call_and_return_conditional_losses_309057�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
,:*��2conv2d_25/kernel
:�2conv2d_25/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_26_layer_call_fn_309066�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_26_layer_call_and_return_conditional_losses_309077�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
,:*��2conv2d_26/kernel
:�2conv2d_26/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
3__inference_conv2d_transpose_5_layer_call_fn_309086�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_309119�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
4:2@�2conv2d_transpose_5/kernel
%:#@2conv2d_transpose_5/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_concatenate_5_layer_call_fn_309125�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_concatenate_5_layer_call_and_return_conditional_losses_309132�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_27_layer_call_fn_309141�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_27_layer_call_and_return_conditional_losses_309152�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
+:)�@2conv2d_27/kernel
:@2conv2d_27/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_28_layer_call_fn_309161�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_28_layer_call_and_return_conditional_losses_309172�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
*:(@@2conv2d_28/kernel
:@2conv2d_28/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_29_layer_call_fn_309181�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_29_layer_call_and_return_conditional_losses_309192�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
*:(@2conv2d_29/kernel
:2conv2d_29/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_model_1_layer_call_fn_307610input_2"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
(__inference_model_1_layer_call_fn_307787input_2"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
(__inference_model_1_layer_call_fn_308270inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
(__inference_model_1_layer_call_fn_308347inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_model_1_layer_call_and_return_conditional_losses_307332input_2"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_model_1_layer_call_and_return_conditional_losses_307432input_2"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_model_1_layer_call_and_return_conditional_losses_308522inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_model_1_layer_call_and_return_conditional_losses_308697inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56
�57
�58
�59
�60
�61
�62
�63
�64
�65
�66
�67
�68
�69
�70
�71
�72"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35"
trackable_list_wrapper
�
�trace_0
�trace_1
�trace_2
�trace_3
�trace_4
�trace_5
�trace_6
�trace_7
�trace_8
�trace_9
�trace_10
�trace_11
�trace_12
�trace_13
�trace_14
�trace_15
�trace_16
�trace_17
�trace_18
�trace_19
�trace_20
�trace_21
�trace_22
�trace_23
�trace_24
�trace_25
�trace_26
�trace_27
�trace_28
�trace_29
�trace_30
�trace_31
�trace_32
�trace_33
�trace_34
�trace_352�
#__inference__update_step_xla_297815
#__inference__update_step_xla_297820
#__inference__update_step_xla_297825
#__inference__update_step_xla_297830
#__inference__update_step_xla_297835
#__inference__update_step_xla_297840
#__inference__update_step_xla_297845
#__inference__update_step_xla_297850
#__inference__update_step_xla_297855
#__inference__update_step_xla_297860
#__inference__update_step_xla_297865
#__inference__update_step_xla_297870
#__inference__update_step_xla_297875
#__inference__update_step_xla_297880
#__inference__update_step_xla_297885
#__inference__update_step_xla_297890
#__inference__update_step_xla_297895
#__inference__update_step_xla_297900
#__inference__update_step_xla_297905
#__inference__update_step_xla_297910
#__inference__update_step_xla_297915
#__inference__update_step_xla_297920
#__inference__update_step_xla_297925
#__inference__update_step_xla_297930
#__inference__update_step_xla_297935
#__inference__update_step_xla_297940
#__inference__update_step_xla_297945
#__inference__update_step_xla_297950
#__inference__update_step_xla_297955
#__inference__update_step_xla_297960
#__inference__update_step_xla_297965
#__inference__update_step_xla_297970
#__inference__update_step_xla_297975
#__inference__update_step_xla_297980
#__inference__update_step_xla_297985
#__inference__update_step_xla_297990�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0z�trace_0z�trace_1z�trace_2z�trace_3z�trace_4z�trace_5z�trace_6z�trace_7z�trace_8z�trace_9z�trace_10z�trace_11z�trace_12z�trace_13z�trace_14z�trace_15z�trace_16z�trace_17z�trace_18z�trace_19z�trace_20z�trace_21z�trace_22z�trace_23z�trace_24z�trace_25z�trace_26z�trace_27z�trace_28z�trace_29z�trace_30z�trace_31z�trace_32z�trace_33z�trace_34z�trace_35
�B�
$__inference_signature_wrapper_308193input_2"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_conv2d_15_layer_call_fn_308706inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv2d_15_layer_call_and_return_conditional_losses_308717inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_conv2d_16_layer_call_fn_308726inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv2d_16_layer_call_and_return_conditional_losses_308737inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
0__inference_max_pooling2d_3_layer_call_fn_308742inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_308747inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_conv2d_17_layer_call_fn_308756inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv2d_17_layer_call_and_return_conditional_losses_308767inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_conv2d_18_layer_call_fn_308776inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv2d_18_layer_call_and_return_conditional_losses_308787inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
0__inference_max_pooling2d_4_layer_call_fn_308792inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_308797inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_conv2d_19_layer_call_fn_308806inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv2d_19_layer_call_and_return_conditional_losses_308817inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_conv2d_20_layer_call_fn_308826inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv2d_20_layer_call_and_return_conditional_losses_308837inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
0__inference_max_pooling2d_5_layer_call_fn_308842inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_308847inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_conv2d_21_layer_call_fn_308856inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv2d_21_layer_call_and_return_conditional_losses_308867inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_conv2d_22_layer_call_fn_308876inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv2d_22_layer_call_and_return_conditional_losses_308887inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
3__inference_conv2d_transpose_3_layer_call_fn_308896inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_308929inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_concatenate_3_layer_call_fn_308935inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_concatenate_3_layer_call_and_return_conditional_losses_308942inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_conv2d_23_layer_call_fn_308951inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv2d_23_layer_call_and_return_conditional_losses_308962inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_conv2d_24_layer_call_fn_308971inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv2d_24_layer_call_and_return_conditional_losses_308982inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
3__inference_conv2d_transpose_4_layer_call_fn_308991inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_309024inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_concatenate_4_layer_call_fn_309030inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_concatenate_4_layer_call_and_return_conditional_losses_309037inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_conv2d_25_layer_call_fn_309046inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv2d_25_layer_call_and_return_conditional_losses_309057inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_conv2d_26_layer_call_fn_309066inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv2d_26_layer_call_and_return_conditional_losses_309077inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
3__inference_conv2d_transpose_5_layer_call_fn_309086inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_309119inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_concatenate_5_layer_call_fn_309125inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_concatenate_5_layer_call_and_return_conditional_losses_309132inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_conv2d_27_layer_call_fn_309141inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv2d_27_layer_call_and_return_conditional_losses_309152inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_conv2d_28_layer_call_fn_309161inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv2d_28_layer_call_and_return_conditional_losses_309172inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_conv2d_29_layer_call_fn_309181inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv2d_29_layer_call_and_return_conditional_losses_309192inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
/:-@2Adam/m/conv2d_15/kernel
/:-@2Adam/v/conv2d_15/kernel
!:@2Adam/m/conv2d_15/bias
!:@2Adam/v/conv2d_15/bias
/:-@@2Adam/m/conv2d_16/kernel
/:-@@2Adam/v/conv2d_16/kernel
!:@2Adam/m/conv2d_16/bias
!:@2Adam/v/conv2d_16/bias
0:.@�2Adam/m/conv2d_17/kernel
0:.@�2Adam/v/conv2d_17/kernel
": �2Adam/m/conv2d_17/bias
": �2Adam/v/conv2d_17/bias
1:/��2Adam/m/conv2d_18/kernel
1:/��2Adam/v/conv2d_18/kernel
": �2Adam/m/conv2d_18/bias
": �2Adam/v/conv2d_18/bias
1:/��2Adam/m/conv2d_19/kernel
1:/��2Adam/v/conv2d_19/kernel
": �2Adam/m/conv2d_19/bias
": �2Adam/v/conv2d_19/bias
1:/��2Adam/m/conv2d_20/kernel
1:/��2Adam/v/conv2d_20/kernel
": �2Adam/m/conv2d_20/bias
": �2Adam/v/conv2d_20/bias
1:/��2Adam/m/conv2d_21/kernel
1:/��2Adam/v/conv2d_21/kernel
": �2Adam/m/conv2d_21/bias
": �2Adam/v/conv2d_21/bias
1:/��2Adam/m/conv2d_22/kernel
1:/��2Adam/v/conv2d_22/kernel
": �2Adam/m/conv2d_22/bias
": �2Adam/v/conv2d_22/bias
::8��2 Adam/m/conv2d_transpose_3/kernel
::8��2 Adam/v/conv2d_transpose_3/kernel
+:)�2Adam/m/conv2d_transpose_3/bias
+:)�2Adam/v/conv2d_transpose_3/bias
1:/��2Adam/m/conv2d_23/kernel
1:/��2Adam/v/conv2d_23/kernel
": �2Adam/m/conv2d_23/bias
": �2Adam/v/conv2d_23/bias
1:/��2Adam/m/conv2d_24/kernel
1:/��2Adam/v/conv2d_24/kernel
": �2Adam/m/conv2d_24/bias
": �2Adam/v/conv2d_24/bias
::8��2 Adam/m/conv2d_transpose_4/kernel
::8��2 Adam/v/conv2d_transpose_4/kernel
+:)�2Adam/m/conv2d_transpose_4/bias
+:)�2Adam/v/conv2d_transpose_4/bias
1:/��2Adam/m/conv2d_25/kernel
1:/��2Adam/v/conv2d_25/kernel
": �2Adam/m/conv2d_25/bias
": �2Adam/v/conv2d_25/bias
1:/��2Adam/m/conv2d_26/kernel
1:/��2Adam/v/conv2d_26/kernel
": �2Adam/m/conv2d_26/bias
": �2Adam/v/conv2d_26/bias
9:7@�2 Adam/m/conv2d_transpose_5/kernel
9:7@�2 Adam/v/conv2d_transpose_5/kernel
*:(@2Adam/m/conv2d_transpose_5/bias
*:(@2Adam/v/conv2d_transpose_5/bias
0:.�@2Adam/m/conv2d_27/kernel
0:.�@2Adam/v/conv2d_27/kernel
!:@2Adam/m/conv2d_27/bias
!:@2Adam/v/conv2d_27/bias
/:-@@2Adam/m/conv2d_28/kernel
/:-@@2Adam/v/conv2d_28/kernel
!:@2Adam/m/conv2d_28/bias
!:@2Adam/v/conv2d_28/bias
/:-@2Adam/m/conv2d_29/kernel
/:-@2Adam/v/conv2d_29/kernel
!:2Adam/m/conv2d_29/bias
!:2Adam/v/conv2d_29/bias
�B�
#__inference__update_step_xla_297815gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_297820gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_297825gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_297830gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_297835gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_297840gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_297845gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_297850gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_297855gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_297860gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_297865gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_297870gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_297875gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_297880gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_297885gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_297890gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_297895gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_297900gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_297905gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_297910gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_297915gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_297920gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_297925gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_297930gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_297935gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_297940gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_297945gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_297950gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_297955gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_297960gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_297965gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_297970gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_297975gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_297980gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_297985gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_297990gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper�
#__inference__update_step_xla_297815~x�u
n�k
!�
gradient@
<�9	%�"
�@
�
p
` VariableSpec 
`������?
� "
 �
#__inference__update_step_xla_297820f`�]
V�S
�
gradient@
0�-	�
�@
�
p
` VariableSpec 
`������?
� "
 �
#__inference__update_step_xla_297825~x�u
n�k
!�
gradient@@
<�9	%�"
�@@
�
p
` VariableSpec 
`������?
� "
 �
#__inference__update_step_xla_297830f`�]
V�S
�
gradient@
0�-	�
�@
�
p
` VariableSpec 
`������?
� "
 �
#__inference__update_step_xla_297835�z�w
p�m
"�
gradient@�
=�:	&�#
�@�
�
p
` VariableSpec 
`�饁��?
� "
 �
#__inference__update_step_xla_297840hb�_
X�U
�
gradient�
1�.	�
��
�
p
` VariableSpec 
`�襁��?
� "
 �
#__inference__update_step_xla_297845�|�y
r�o
#� 
gradient��
>�;	'�$
���
�
p
` VariableSpec 
`������?
� "
 �
#__inference__update_step_xla_297850hb�_
X�U
�
gradient�
1�.	�
��
�
p
` VariableSpec 
`������?
� "
 �
#__inference__update_step_xla_297855�|�y
r�o
#� 
gradient��
>�;	'�$
���
�
p
` VariableSpec 
`�Ӵ���?
� "
 �
#__inference__update_step_xla_297860hb�_
X�U
�
gradient�
1�.	�
��
�
p
` VariableSpec 
`�Ҵ���?
� "
 �
#__inference__update_step_xla_297865�|�y
r�o
#� 
gradient��
>�;	'�$
���
�
p
` VariableSpec 
`������?
� "
 �
#__inference__update_step_xla_297870hb�_
X�U
�
gradient�
1�.	�
��
�
p
` VariableSpec 
`������?
� "
 �
#__inference__update_step_xla_297875�|�y
r�o
#� 
gradient��
>�;	'�$
���
�
p
` VariableSpec 
`�끁��?
� "
 �
#__inference__update_step_xla_297880hb�_
X�U
�
gradient�
1�.	�
��
�
p
` VariableSpec 
`������?
� "
 �
#__inference__update_step_xla_297885�|�y
r�o
#� 
gradient��
>�;	'�$
���
�
p
` VariableSpec 
`�ڀ���?
� "
 �
#__inference__update_step_xla_297890hb�_
X�U
�
gradient�
1�.	�
��
�
p
` VariableSpec 
`������?
� "
 �
#__inference__update_step_xla_297895�|�y
r�o
#� 
gradient��
>�;	'�$
���
�
p
` VariableSpec 
`�����?
� "
 �
#__inference__update_step_xla_297900hb�_
X�U
�
gradient�
1�.	�
��
�
p
` VariableSpec 
`�����?
� "
 �
#__inference__update_step_xla_297905�|�y
r�o
#� 
gradient��
>�;	'�$
���
�
p
` VariableSpec 
`�����?
� "
 �
#__inference__update_step_xla_297910hb�_
X�U
�
gradient�
1�.	�
��
�
p
` VariableSpec 
`�����?
� "
 �
#__inference__update_step_xla_297915�|�y
r�o
#� 
gradient��
>�;	'�$
���
�
p
` VariableSpec 
`�����?
� "
 �
#__inference__update_step_xla_297920hb�_
X�U
�
gradient�
1�.	�
��
�
p
` VariableSpec 
`�����?
� "
 �
#__inference__update_step_xla_297925�|�y
r�o
#� 
gradient��
>�;	'�$
���
�
p
` VariableSpec 
`������?
� "
 �
#__inference__update_step_xla_297930hb�_
X�U
�
gradient�
1�.	�
��
�
p
` VariableSpec 
`������?
� "
 �
#__inference__update_step_xla_297935�|�y
r�o
#� 
gradient��
>�;	'�$
���
�
p
` VariableSpec 
`�ֵ���?
� "
 �
#__inference__update_step_xla_297940hb�_
X�U
�
gradient�
1�.	�
��
�
p
` VariableSpec 
`�յ���?
� "
 �
#__inference__update_step_xla_297945�|�y
r�o
#� 
gradient��
>�;	'�$
���
�
p
` VariableSpec 
`������?
� "
 �
#__inference__update_step_xla_297950hb�_
X�U
�
gradient�
1�.	�
��
�
p
` VariableSpec 
`������?
� "
 �
#__inference__update_step_xla_297955�z�w
p�m
"�
gradient@�
=�:	&�#
�@�
�
p
` VariableSpec 
`������?
� "
 �
#__inference__update_step_xla_297960f`�]
V�S
�
gradient@
0�-	�
�@
�
p
` VariableSpec 
`ਾ���?
� "
 �
#__inference__update_step_xla_297965�z�w
p�m
"�
gradient�@
=�:	&�#
��@
�
p
` VariableSpec 
`�����?
� "
 �
#__inference__update_step_xla_297970f`�]
V�S
�
gradient@
0�-	�
�@
�
p
` VariableSpec 
`�����?
� "
 �
#__inference__update_step_xla_297975~x�u
n�k
!�
gradient@@
<�9	%�"
�@@
�
p
` VariableSpec 
`�ؾ���?
� "
 �
#__inference__update_step_xla_297980f`�]
V�S
�
gradient@
0�-	�
�@
�
p
` VariableSpec 
`������?
� "
 �
#__inference__update_step_xla_297985~x�u
n�k
!�
gradient@
<�9	%�"
�@
�
p
` VariableSpec 
`�˿���?
� "
 �
#__inference__update_step_xla_297990f`�]
V�S
�
gradient
0�-	�
�
�
p
` VariableSpec 
`�ʿ���?
� "
 �
!__inference__wrapped_model_306859�8)*23ABJKYZbcqrz{��������������������8�5
.�+
)�&
input_2���������@@
� "=�:
8
	conv2d_29+�(
	conv2d_29���������@@�
I__inference_concatenate_3_layer_call_and_return_conditional_losses_308942�l�i
b�_
]�Z
+�(
inputs_0����������
+�(
inputs_1����������
� "5�2
+�(
tensor_0����������
� �
.__inference_concatenate_3_layer_call_fn_308935�l�i
b�_
]�Z
+�(
inputs_0����������
+�(
inputs_1����������
� "*�'
unknown�����������
I__inference_concatenate_4_layer_call_and_return_conditional_losses_309037�l�i
b�_
]�Z
+�(
inputs_0���������  �
+�(
inputs_1���������  �
� "5�2
+�(
tensor_0���������  �
� �
.__inference_concatenate_4_layer_call_fn_309030�l�i
b�_
]�Z
+�(
inputs_0���������  �
+�(
inputs_1���������  �
� "*�'
unknown���������  ��
I__inference_concatenate_5_layer_call_and_return_conditional_losses_309132�j�g
`�]
[�X
*�'
inputs_0���������@@@
*�'
inputs_1���������@@@
� "5�2
+�(
tensor_0���������@@�
� �
.__inference_concatenate_5_layer_call_fn_309125�j�g
`�]
[�X
*�'
inputs_0���������@@@
*�'
inputs_1���������@@@
� "*�'
unknown���������@@��
E__inference_conv2d_15_layer_call_and_return_conditional_losses_308717s)*7�4
-�*
(�%
inputs���������@@
� "4�1
*�'
tensor_0���������@@@
� �
*__inference_conv2d_15_layer_call_fn_308706h)*7�4
-�*
(�%
inputs���������@@
� ")�&
unknown���������@@@�
E__inference_conv2d_16_layer_call_and_return_conditional_losses_308737s237�4
-�*
(�%
inputs���������@@@
� "4�1
*�'
tensor_0���������@@@
� �
*__inference_conv2d_16_layer_call_fn_308726h237�4
-�*
(�%
inputs���������@@@
� ")�&
unknown���������@@@�
E__inference_conv2d_17_layer_call_and_return_conditional_losses_308767tAB7�4
-�*
(�%
inputs���������  @
� "5�2
+�(
tensor_0���������  �
� �
*__inference_conv2d_17_layer_call_fn_308756iAB7�4
-�*
(�%
inputs���������  @
� "*�'
unknown���������  ��
E__inference_conv2d_18_layer_call_and_return_conditional_losses_308787uJK8�5
.�+
)�&
inputs���������  �
� "5�2
+�(
tensor_0���������  �
� �
*__inference_conv2d_18_layer_call_fn_308776jJK8�5
.�+
)�&
inputs���������  �
� "*�'
unknown���������  ��
E__inference_conv2d_19_layer_call_and_return_conditional_losses_308817uYZ8�5
.�+
)�&
inputs����������
� "5�2
+�(
tensor_0����������
� �
*__inference_conv2d_19_layer_call_fn_308806jYZ8�5
.�+
)�&
inputs����������
� "*�'
unknown�����������
E__inference_conv2d_20_layer_call_and_return_conditional_losses_308837ubc8�5
.�+
)�&
inputs����������
� "5�2
+�(
tensor_0����������
� �
*__inference_conv2d_20_layer_call_fn_308826jbc8�5
.�+
)�&
inputs����������
� "*�'
unknown�����������
E__inference_conv2d_21_layer_call_and_return_conditional_losses_308867uqr8�5
.�+
)�&
inputs����������
� "5�2
+�(
tensor_0����������
� �
*__inference_conv2d_21_layer_call_fn_308856jqr8�5
.�+
)�&
inputs����������
� "*�'
unknown�����������
E__inference_conv2d_22_layer_call_and_return_conditional_losses_308887uz{8�5
.�+
)�&
inputs����������
� "5�2
+�(
tensor_0����������
� �
*__inference_conv2d_22_layer_call_fn_308876jz{8�5
.�+
)�&
inputs����������
� "*�'
unknown�����������
E__inference_conv2d_23_layer_call_and_return_conditional_losses_308962w��8�5
.�+
)�&
inputs����������
� "5�2
+�(
tensor_0����������
� �
*__inference_conv2d_23_layer_call_fn_308951l��8�5
.�+
)�&
inputs����������
� "*�'
unknown�����������
E__inference_conv2d_24_layer_call_and_return_conditional_losses_308982w��8�5
.�+
)�&
inputs����������
� "5�2
+�(
tensor_0����������
� �
*__inference_conv2d_24_layer_call_fn_308971l��8�5
.�+
)�&
inputs����������
� "*�'
unknown�����������
E__inference_conv2d_25_layer_call_and_return_conditional_losses_309057w��8�5
.�+
)�&
inputs���������  �
� "5�2
+�(
tensor_0���������  �
� �
*__inference_conv2d_25_layer_call_fn_309046l��8�5
.�+
)�&
inputs���������  �
� "*�'
unknown���������  ��
E__inference_conv2d_26_layer_call_and_return_conditional_losses_309077w��8�5
.�+
)�&
inputs���������  �
� "5�2
+�(
tensor_0���������  �
� �
*__inference_conv2d_26_layer_call_fn_309066l��8�5
.�+
)�&
inputs���������  �
� "*�'
unknown���������  ��
E__inference_conv2d_27_layer_call_and_return_conditional_losses_309152v��8�5
.�+
)�&
inputs���������@@�
� "4�1
*�'
tensor_0���������@@@
� �
*__inference_conv2d_27_layer_call_fn_309141k��8�5
.�+
)�&
inputs���������@@�
� ")�&
unknown���������@@@�
E__inference_conv2d_28_layer_call_and_return_conditional_losses_309172u��7�4
-�*
(�%
inputs���������@@@
� "4�1
*�'
tensor_0���������@@@
� �
*__inference_conv2d_28_layer_call_fn_309161j��7�4
-�*
(�%
inputs���������@@@
� ")�&
unknown���������@@@�
E__inference_conv2d_29_layer_call_and_return_conditional_losses_309192u��7�4
-�*
(�%
inputs���������@@@
� "4�1
*�'
tensor_0���������@@
� �
*__inference_conv2d_29_layer_call_fn_309181j��7�4
-�*
(�%
inputs���������@@@
� ")�&
unknown���������@@�
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_308929���J�G
@�=
;�8
inputs,����������������������������
� "G�D
=�:
tensor_0,����������������������������
� �
3__inference_conv2d_transpose_3_layer_call_fn_308896���J�G
@�=
;�8
inputs,����������������������������
� "<�9
unknown,�����������������������������
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_309024���J�G
@�=
;�8
inputs,����������������������������
� "G�D
=�:
tensor_0,����������������������������
� �
3__inference_conv2d_transpose_4_layer_call_fn_308991���J�G
@�=
;�8
inputs,����������������������������
� "<�9
unknown,�����������������������������
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_309119���J�G
@�=
;�8
inputs,����������������������������
� "F�C
<�9
tensor_0+���������������������������@
� �
3__inference_conv2d_transpose_5_layer_call_fn_309086���J�G
@�=
;�8
inputs,����������������������������
� ";�8
unknown+���������������������������@�
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_308747�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
0__inference_max_pooling2d_3_layer_call_fn_308742�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_308797�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
0__inference_max_pooling2d_4_layer_call_fn_308792�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
K__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_308847�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
0__inference_max_pooling2d_5_layer_call_fn_308842�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
C__inference_model_1_layer_call_and_return_conditional_losses_307332�8)*23ABJKYZbcqrz{��������������������@�=
6�3
)�&
input_2���������@@
p

 
� "4�1
*�'
tensor_0���������@@
� �
C__inference_model_1_layer_call_and_return_conditional_losses_307432�8)*23ABJKYZbcqrz{��������������������@�=
6�3
)�&
input_2���������@@
p 

 
� "4�1
*�'
tensor_0���������@@
� �
C__inference_model_1_layer_call_and_return_conditional_losses_308522�8)*23ABJKYZbcqrz{��������������������?�<
5�2
(�%
inputs���������@@
p

 
� "4�1
*�'
tensor_0���������@@
� �
C__inference_model_1_layer_call_and_return_conditional_losses_308697�8)*23ABJKYZbcqrz{��������������������?�<
5�2
(�%
inputs���������@@
p 

 
� "4�1
*�'
tensor_0���������@@
� �
(__inference_model_1_layer_call_fn_307610�8)*23ABJKYZbcqrz{��������������������@�=
6�3
)�&
input_2���������@@
p

 
� ")�&
unknown���������@@�
(__inference_model_1_layer_call_fn_307787�8)*23ABJKYZbcqrz{��������������������@�=
6�3
)�&
input_2���������@@
p 

 
� ")�&
unknown���������@@�
(__inference_model_1_layer_call_fn_308270�8)*23ABJKYZbcqrz{��������������������?�<
5�2
(�%
inputs���������@@
p

 
� ")�&
unknown���������@@�
(__inference_model_1_layer_call_fn_308347�8)*23ABJKYZbcqrz{��������������������?�<
5�2
(�%
inputs���������@@
p 

 
� ")�&
unknown���������@@�
$__inference_signature_wrapper_308193�8)*23ABJKYZbcqrz{��������������������C�@
� 
9�6
4
input_2)�&
input_2���������@@"=�:
8
	conv2d_29+�(
	conv2d_29���������@@