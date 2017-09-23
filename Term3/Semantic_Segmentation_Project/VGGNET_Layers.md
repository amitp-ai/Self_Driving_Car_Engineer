Operations:  save/RestoreV2_1/tensor_names
save/RestoreV2_1/tensor_names Output:  save/RestoreV2_1/tensor_names:0 (1,)

Operations:  save/RestoreV2_1/shape_and_slices
save/RestoreV2_1/shape_and_slices Output:  save/RestoreV2_1/shape_and_slices:0 (1,)

Operations:  save/RestoreV2_1
save/RestoreV2_1 Input:  save/Const:0 ()
save/RestoreV2_1 Input:  save/RestoreV2_1/tensor_names:0 (1,)
save/RestoreV2_1 Input:  save/RestoreV2_1/shape_and_slices:0 (1,)
save/RestoreV2_1 Output:  save/RestoreV2_1:0 <unknown>

Operations:  save/Assign_1
save/Assign_1 Input:  conv1_1/filter:0 (3, 3, 3, 64)
save/Assign_1 Input:  save/RestoreV2_1:0 <unknown>
save/Assign_1 Output:  save/Assign_1:0 (3, 3, 3, 64)

Operations:  save/RestoreV2_2/tensor_names
save/RestoreV2_2/tensor_names Output:  save/RestoreV2_2/tensor_names:0 (1,)

Operations:  save/RestoreV2_2/shape_and_slices
save/RestoreV2_2/shape_and_slices Output:  save/RestoreV2_2/shape_and_slices:0 (1,)

Operations:  save/RestoreV2_2
save/RestoreV2_2 Input:  save/Const:0 ()
save/RestoreV2_2 Input:  save/RestoreV2_2/tensor_names:0 (1,)
save/RestoreV2_2 Input:  save/RestoreV2_2/shape_and_slices:0 (1,)
save/RestoreV2_2 Output:  save/RestoreV2_2:0 <unknown>

Operations:  save/Assign_2
save/Assign_2 Input:  conv1_2/biases:0 (64,)
save/Assign_2 Input:  save/RestoreV2_2:0 <unknown>
save/Assign_2 Output:  save/Assign_2:0 (64,)

Operations:  save/RestoreV2_3/tensor_names
save/RestoreV2_3/tensor_names Output:  save/RestoreV2_3/tensor_names:0 (1,)

Operations:  save/RestoreV2_3/shape_and_slices
save/RestoreV2_3/shape_and_slices Output:  save/RestoreV2_3/shape_and_slices:0 (1,)

Operations:  save/RestoreV2_3
save/RestoreV2_3 Input:  save/Const:0 ()
save/RestoreV2_3 Input:  save/RestoreV2_3/tensor_names:0 (1,)
save/RestoreV2_3 Input:  save/RestoreV2_3/shape_and_slices:0 (1,)
save/RestoreV2_3 Output:  save/RestoreV2_3:0 <unknown>

Operations:  save/Assign_3
save/Assign_3 Input:  conv1_2/filter:0 (3, 3, 64, 64)
save/Assign_3 Input:  save/RestoreV2_3:0 <unknown>
save/Assign_3 Output:  save/Assign_3:0 (3, 3, 64, 64)

Operations:  save/RestoreV2_4/tensor_names
save/RestoreV2_4/tensor_names Output:  save/RestoreV2_4/tensor_names:0 (1,)

Operations:  save/RestoreV2_4/shape_and_slices
save/RestoreV2_4/shape_and_slices Output:  save/RestoreV2_4/shape_and_slices:0 (1,)

Operations:  save/RestoreV2_4
save/RestoreV2_4 Input:  save/Const:0 ()
save/RestoreV2_4 Input:  save/RestoreV2_4/tensor_names:0 (1,)
save/RestoreV2_4 Input:  save/RestoreV2_4/shape_and_slices:0 (1,)
save/RestoreV2_4 Output:  save/RestoreV2_4:0 <unknown>

Operations:  save/Assign_4
save/Assign_4 Input:  conv2_1/biases:0 (128,)
save/Assign_4 Input:  save/RestoreV2_4:0 <unknown>
save/Assign_4 Output:  save/Assign_4:0 (128,)

Operations:  save/RestoreV2_5/tensor_names
save/RestoreV2_5/tensor_names Output:  save/RestoreV2_5/tensor_names:0 (1,)

Operations:  save/RestoreV2_5/shape_and_slices
save/RestoreV2_5/shape_and_slices Output:  save/RestoreV2_5/shape_and_slices:0 (1,)

Operations:  save/RestoreV2_5
save/RestoreV2_5 Input:  save/Const:0 ()
save/RestoreV2_5 Input:  save/RestoreV2_5/tensor_names:0 (1,)
save/RestoreV2_5 Input:  save/RestoreV2_5/shape_and_slices:0 (1,)
save/RestoreV2_5 Output:  save/RestoreV2_5:0 <unknown>

Operations:  save/Assign_5
save/Assign_5 Input:  conv2_1/filter:0 (3, 3, 64, 128)
save/Assign_5 Input:  save/RestoreV2_5:0 <unknown>
save/Assign_5 Output:  save/Assign_5:0 (3, 3, 64, 128)

Operations:  save/RestoreV2_6/tensor_names
save/RestoreV2_6/tensor_names Output:  save/RestoreV2_6/tensor_names:0 (1,)

Operations:  save/RestoreV2_6/shape_and_slices
save/RestoreV2_6/shape_and_slices Output:  save/RestoreV2_6/shape_and_slices:0 (1,)

Operations:  save/RestoreV2_6
save/RestoreV2_6 Input:  save/Const:0 ()
save/RestoreV2_6 Input:  save/RestoreV2_6/tensor_names:0 (1,)
save/RestoreV2_6 Input:  save/RestoreV2_6/shape_and_slices:0 (1,)
save/RestoreV2_6 Output:  save/RestoreV2_6:0 <unknown>

Operations:  save/Assign_6
save/Assign_6 Input:  conv2_2/biases:0 (128,)
save/Assign_6 Input:  save/RestoreV2_6:0 <unknown>
save/Assign_6 Output:  save/Assign_6:0 (128,)

Operations:  save/RestoreV2_7/tensor_names
save/RestoreV2_7/tensor_names Output:  save/RestoreV2_7/tensor_names:0 (1,)

Operations:  save/RestoreV2_7/shape_and_slices
save/RestoreV2_7/shape_and_slices Output:  save/RestoreV2_7/shape_and_slices:0 (1,)

Operations:  save/RestoreV2_7
save/RestoreV2_7 Input:  save/Const:0 ()
save/RestoreV2_7 Input:  save/RestoreV2_7/tensor_names:0 (1,)
save/RestoreV2_7 Input:  save/RestoreV2_7/shape_and_slices:0 (1,)
save/RestoreV2_7 Output:  save/RestoreV2_7:0 <unknown>

Operations:  save/Assign_7
save/Assign_7 Input:  conv2_2/filter:0 (3, 3, 128, 128)
save/Assign_7 Input:  save/RestoreV2_7:0 <unknown>
save/Assign_7 Output:  save/Assign_7:0 (3, 3, 128, 128)

Operations:  save/RestoreV2_8/tensor_names
save/RestoreV2_8/tensor_names Output:  save/RestoreV2_8/tensor_names:0 (1,)

Operations:  save/RestoreV2_8/shape_and_slices
save/RestoreV2_8/shape_and_slices Output:  save/RestoreV2_8/shape_and_slices:0 (1,)

Operations:  save/RestoreV2_8
save/RestoreV2_8 Input:  save/Const:0 ()
save/RestoreV2_8 Input:  save/RestoreV2_8/tensor_names:0 (1,)
save/RestoreV2_8 Input:  save/RestoreV2_8/shape_and_slices:0 (1,)
save/RestoreV2_8 Output:  save/RestoreV2_8:0 <unknown>

Operations:  save/Assign_8
save/Assign_8 Input:  conv3_1/biases:0 (256,)
save/Assign_8 Input:  save/RestoreV2_8:0 <unknown>
save/Assign_8 Output:  save/Assign_8:0 (256,)

Operations:  save/RestoreV2_9/tensor_names
save/RestoreV2_9/tensor_names Output:  save/RestoreV2_9/tensor_names:0 (1,)

Operations:  save/RestoreV2_9/shape_and_slices
save/RestoreV2_9/shape_and_slices Output:  save/RestoreV2_9/shape_and_slices:0 (1,)

Operations:  save/RestoreV2_9
save/RestoreV2_9 Input:  save/Const:0 ()
save/RestoreV2_9 Input:  save/RestoreV2_9/tensor_names:0 (1,)
save/RestoreV2_9 Input:  save/RestoreV2_9/shape_and_slices:0 (1,)
save/RestoreV2_9 Output:  save/RestoreV2_9:0 <unknown>

Operations:  save/Assign_9
save/Assign_9 Input:  conv3_1/filter:0 (3, 3, 128, 256)
save/Assign_9 Input:  save/RestoreV2_9:0 <unknown>
save/Assign_9 Output:  save/Assign_9:0 (3, 3, 128, 256)

Operations:  save/RestoreV2_10/tensor_names
save/RestoreV2_10/tensor_names Output:  save/RestoreV2_10/tensor_names:0 (1,)

Operations:  save/RestoreV2_10/shape_and_slices
save/RestoreV2_10/shape_and_slices Output:  save/RestoreV2_10/shape_and_slices:0 (1,)

Operations:  save/RestoreV2_10
save/RestoreV2_10 Input:  save/Const:0 ()
save/RestoreV2_10 Input:  save/RestoreV2_10/tensor_names:0 (1,)
save/RestoreV2_10 Input:  save/RestoreV2_10/shape_and_slices:0 (1,)
save/RestoreV2_10 Output:  save/RestoreV2_10:0 <unknown>

Operations:  save/Assign_10
save/Assign_10 Input:  conv3_2/biases:0 (256,)
save/Assign_10 Input:  save/RestoreV2_10:0 <unknown>
save/Assign_10 Output:  save/Assign_10:0 (256,)

Operations:  save/RestoreV2_11/tensor_names
save/RestoreV2_11/tensor_names Output:  save/RestoreV2_11/tensor_names:0 (1,)

Operations:  save/RestoreV2_11/shape_and_slices
save/RestoreV2_11/shape_and_slices Output:  save/RestoreV2_11/shape_and_slices:0 (1,)

Operations:  save/RestoreV2_11
save/RestoreV2_11 Input:  save/Const:0 ()
save/RestoreV2_11 Input:  save/RestoreV2_11/tensor_names:0 (1,)
save/RestoreV2_11 Input:  save/RestoreV2_11/shape_and_slices:0 (1,)
save/RestoreV2_11 Output:  save/RestoreV2_11:0 <unknown>

Operations:  save/Assign_11
save/Assign_11 Input:  conv3_2/filter:0 (3, 3, 256, 256)
save/Assign_11 Input:  save/RestoreV2_11:0 <unknown>
save/Assign_11 Output:  save/Assign_11:0 (3, 3, 256, 256)

Operations:  save/RestoreV2_12/tensor_names
save/RestoreV2_12/tensor_names Output:  save/RestoreV2_12/tensor_names:0 (1,)

Operations:  save/RestoreV2_12/shape_and_slices
save/RestoreV2_12/shape_and_slices Output:  save/RestoreV2_12/shape_and_slices:0 (1,)

Operations:  save/RestoreV2_12
save/RestoreV2_12 Input:  save/Const:0 ()
save/RestoreV2_12 Input:  save/RestoreV2_12/tensor_names:0 (1,)
save/RestoreV2_12 Input:  save/RestoreV2_12/shape_and_slices:0 (1,)
save/RestoreV2_12 Output:  save/RestoreV2_12:0 <unknown>

Operations:  save/Assign_12
save/Assign_12 Input:  conv3_3/biases:0 (256,)
save/Assign_12 Input:  save/RestoreV2_12:0 <unknown>
save/Assign_12 Output:  save/Assign_12:0 (256,)

Operations:  save/RestoreV2_13/tensor_names
save/RestoreV2_13/tensor_names Output:  save/RestoreV2_13/tensor_names:0 (1,)

Operations:  save/RestoreV2_13/shape_and_slices
save/RestoreV2_13/shape_and_slices Output:  save/RestoreV2_13/shape_and_slices:0 (1,)

Operations:  save/RestoreV2_13
save/RestoreV2_13 Input:  save/Const:0 ()
save/RestoreV2_13 Input:  save/RestoreV2_13/tensor_names:0 (1,)
save/RestoreV2_13 Input:  save/RestoreV2_13/shape_and_slices:0 (1,)
save/RestoreV2_13 Output:  save/RestoreV2_13:0 <unknown>

Operations:  save/Assign_13
save/Assign_13 Input:  conv3_3/filter:0 (3, 3, 256, 256)
save/Assign_13 Input:  save/RestoreV2_13:0 <unknown>
save/Assign_13 Output:  save/Assign_13:0 (3, 3, 256, 256)

Operations:  save/RestoreV2_14/tensor_names
save/RestoreV2_14/tensor_names Output:  save/RestoreV2_14/tensor_names:0 (1,)

Operations:  save/RestoreV2_14/shape_and_slices
save/RestoreV2_14/shape_and_slices Output:  save/RestoreV2_14/shape_and_slices:0 (1,)

Operations:  save/RestoreV2_14
save/RestoreV2_14 Input:  save/Const:0 ()
save/RestoreV2_14 Input:  save/RestoreV2_14/tensor_names:0 (1,)
save/RestoreV2_14 Input:  save/RestoreV2_14/shape_and_slices:0 (1,)
save/RestoreV2_14 Output:  save/RestoreV2_14:0 <unknown>

Operations:  save/Assign_14
save/Assign_14 Input:  conv4_1/biases:0 (512,)
save/Assign_14 Input:  save/RestoreV2_14:0 <unknown>
save/Assign_14 Output:  save/Assign_14:0 (512,)

Operations:  save/RestoreV2_15/tensor_names
save/RestoreV2_15/tensor_names Output:  save/RestoreV2_15/tensor_names:0 (1,)

Operations:  save/RestoreV2_15/shape_and_slices
save/RestoreV2_15/shape_and_slices Output:  save/RestoreV2_15/shape_and_slices:0 (1,)

Operations:  save/RestoreV2_15
save/RestoreV2_15 Input:  save/Const:0 ()
save/RestoreV2_15 Input:  save/RestoreV2_15/tensor_names:0 (1,)
save/RestoreV2_15 Input:  save/RestoreV2_15/shape_and_slices:0 (1,)
save/RestoreV2_15 Output:  save/RestoreV2_15:0 <unknown>

Operations:  save/Assign_15
save/Assign_15 Input:  conv4_1/filter:0 (3, 3, 256, 512)
save/Assign_15 Input:  save/RestoreV2_15:0 <unknown>
save/Assign_15 Output:  save/Assign_15:0 (3, 3, 256, 512)

Operations:  save/RestoreV2_16/tensor_names
save/RestoreV2_16/tensor_names Output:  save/RestoreV2_16/tensor_names:0 (1,)

Operations:  save/RestoreV2_16/shape_and_slices
save/RestoreV2_16/shape_and_slices Output:  save/RestoreV2_16/shape_and_slices:0 (1,)

Operations:  save/RestoreV2_16
save/RestoreV2_16 Input:  save/Const:0 ()
save/RestoreV2_16 Input:  save/RestoreV2_16/tensor_names:0 (1,)
save/RestoreV2_16 Input:  save/RestoreV2_16/shape_and_slices:0 (1,)
save/RestoreV2_16 Output:  save/RestoreV2_16:0 <unknown>

Operations:  save/Assign_16
save/Assign_16 Input:  conv4_2/biases:0 (512,)
save/Assign_16 Input:  save/RestoreV2_16:0 <unknown>
save/Assign_16 Output:  save/Assign_16:0 (512,)

Operations:  save/RestoreV2_17/tensor_names
save/RestoreV2_17/tensor_names Output:  save/RestoreV2_17/tensor_names:0 (1,)

Operations:  save/RestoreV2_17/shape_and_slices
save/RestoreV2_17/shape_and_slices Output:  save/RestoreV2_17/shape_and_slices:0 (1,)

Operations:  save/RestoreV2_17
save/RestoreV2_17 Input:  save/Const:0 ()
save/RestoreV2_17 Input:  save/RestoreV2_17/tensor_names:0 (1,)
save/RestoreV2_17 Input:  save/RestoreV2_17/shape_and_slices:0 (1,)
save/RestoreV2_17 Output:  save/RestoreV2_17:0 <unknown>

Operations:  save/Assign_17
save/Assign_17 Input:  conv4_2/filter:0 (3, 3, 512, 512)
save/Assign_17 Input:  save/RestoreV2_17:0 <unknown>
save/Assign_17 Output:  save/Assign_17:0 (3, 3, 512, 512)

Operations:  save/RestoreV2_18/tensor_names
save/RestoreV2_18/tensor_names Output:  save/RestoreV2_18/tensor_names:0 (1,)

Operations:  save/RestoreV2_18/shape_and_slices
save/RestoreV2_18/shape_and_slices Output:  save/RestoreV2_18/shape_and_slices:0 (1,)

Operations:  save/RestoreV2_18
save/RestoreV2_18 Input:  save/Const:0 ()
save/RestoreV2_18 Input:  save/RestoreV2_18/tensor_names:0 (1,)
save/RestoreV2_18 Input:  save/RestoreV2_18/shape_and_slices:0 (1,)
save/RestoreV2_18 Output:  save/RestoreV2_18:0 <unknown>

Operations:  save/Assign_18
save/Assign_18 Input:  conv4_3/biases:0 (512,)
save/Assign_18 Input:  save/RestoreV2_18:0 <unknown>
save/Assign_18 Output:  save/Assign_18:0 (512,)

Operations:  save/RestoreV2_19/tensor_names
save/RestoreV2_19/tensor_names Output:  save/RestoreV2_19/tensor_names:0 (1,)

Operations:  save/RestoreV2_19/shape_and_slices
save/RestoreV2_19/shape_and_slices Output:  save/RestoreV2_19/shape_and_slices:0 (1,)

Operations:  save/RestoreV2_19
save/RestoreV2_19 Input:  save/Const:0 ()
save/RestoreV2_19 Input:  save/RestoreV2_19/tensor_names:0 (1,)
save/RestoreV2_19 Input:  save/RestoreV2_19/shape_and_slices:0 (1,)
save/RestoreV2_19 Output:  save/RestoreV2_19:0 <unknown>

Operations:  save/Assign_19
save/Assign_19 Input:  conv4_3/filter:0 (3, 3, 512, 512)
save/Assign_19 Input:  save/RestoreV2_19:0 <unknown>
save/Assign_19 Output:  save/Assign_19:0 (3, 3, 512, 512)

Operations:  save/RestoreV2_20/tensor_names
save/RestoreV2_20/tensor_names Output:  save/RestoreV2_20/tensor_names:0 (1,)

Operations:  save/RestoreV2_20/shape_and_slices
save/RestoreV2_20/shape_and_slices Output:  save/RestoreV2_20/shape_and_slices:0 (1,)

Operations:  save/RestoreV2_20
save/RestoreV2_20 Input:  save/Const:0 ()
save/RestoreV2_20 Input:  save/RestoreV2_20/tensor_names:0 (1,)
save/RestoreV2_20 Input:  save/RestoreV2_20/shape_and_slices:0 (1,)
save/RestoreV2_20 Output:  save/RestoreV2_20:0 <unknown>

Operations:  save/Assign_20
save/Assign_20 Input:  conv5_1/biases:0 (512,)
save/Assign_20 Input:  save/RestoreV2_20:0 <unknown>
save/Assign_20 Output:  save/Assign_20:0 (512,)

Operations:  save/RestoreV2_21/tensor_names
save/RestoreV2_21/tensor_names Output:  save/RestoreV2_21/tensor_names:0 (1,)

Operations:  save/RestoreV2_21/shape_and_slices
save/RestoreV2_21/shape_and_slices Output:  save/RestoreV2_21/shape_and_slices:0 (1,)

Operations:  save/RestoreV2_21
save/RestoreV2_21 Input:  save/Const:0 ()
save/RestoreV2_21 Input:  save/RestoreV2_21/tensor_names:0 (1,)
save/RestoreV2_21 Input:  save/RestoreV2_21/shape_and_slices:0 (1,)
save/RestoreV2_21 Output:  save/RestoreV2_21:0 <unknown>

Operations:  save/Assign_21
save/Assign_21 Input:  conv5_1/filter:0 (3, 3, 512, 512)
save/Assign_21 Input:  save/RestoreV2_21:0 <unknown>
save/Assign_21 Output:  save/Assign_21:0 (3, 3, 512, 512)

Operations:  save/RestoreV2_22/tensor_names
save/RestoreV2_22/tensor_names Output:  save/RestoreV2_22/tensor_names:0 (1,)

Operations:  save/RestoreV2_22/shape_and_slices
save/RestoreV2_22/shape_and_slices Output:  save/RestoreV2_22/shape_and_slices:0 (1,)

Operations:  save/RestoreV2_22
save/RestoreV2_22 Input:  save/Const:0 ()
save/RestoreV2_22 Input:  save/RestoreV2_22/tensor_names:0 (1,)
save/RestoreV2_22 Input:  save/RestoreV2_22/shape_and_slices:0 (1,)
save/RestoreV2_22 Output:  save/RestoreV2_22:0 <unknown>

Operations:  save/Assign_22
save/Assign_22 Input:  conv5_2/biases:0 (512,)
save/Assign_22 Input:  save/RestoreV2_22:0 <unknown>
save/Assign_22 Output:  save/Assign_22:0 (512,)

Operations:  save/RestoreV2_23/tensor_names
save/RestoreV2_23/tensor_names Output:  save/RestoreV2_23/tensor_names:0 (1,)

Operations:  save/RestoreV2_23/shape_and_slices
save/RestoreV2_23/shape_and_slices Output:  save/RestoreV2_23/shape_and_slices:0 (1,)

Operations:  save/RestoreV2_23
save/RestoreV2_23 Input:  save/Const:0 ()
save/RestoreV2_23 Input:  save/RestoreV2_23/tensor_names:0 (1,)
save/RestoreV2_23 Input:  save/RestoreV2_23/shape_and_slices:0 (1,)
save/RestoreV2_23 Output:  save/RestoreV2_23:0 <unknown>

Operations:  save/Assign_23
save/Assign_23 Input:  conv5_2/filter:0 (3, 3, 512, 512)
save/Assign_23 Input:  save/RestoreV2_23:0 <unknown>
save/Assign_23 Output:  save/Assign_23:0 (3, 3, 512, 512)

Operations:  save/RestoreV2_24/tensor_names
save/RestoreV2_24/tensor_names Output:  save/RestoreV2_24/tensor_names:0 (1,)

Operations:  save/RestoreV2_24/shape_and_slices
save/RestoreV2_24/shape_and_slices Output:  save/RestoreV2_24/shape_and_slices:0 (1,)

Operations:  save/RestoreV2_24
save/RestoreV2_24 Input:  save/Const:0 ()
save/RestoreV2_24 Input:  save/RestoreV2_24/tensor_names:0 (1,)
save/RestoreV2_24 Input:  save/RestoreV2_24/shape_and_slices:0 (1,)
save/RestoreV2_24 Output:  save/RestoreV2_24:0 <unknown>

Operations:  save/Assign_24
save/Assign_24 Input:  conv5_3/biases:0 (512,)
save/Assign_24 Input:  save/RestoreV2_24:0 <unknown>
save/Assign_24 Output:  save/Assign_24:0 (512,)

Operations:  save/RestoreV2_25/tensor_names
save/RestoreV2_25/tensor_names Output:  save/RestoreV2_25/tensor_names:0 (1,)

Operations:  save/RestoreV2_25/shape_and_slices
save/RestoreV2_25/shape_and_slices Output:  save/RestoreV2_25/shape_and_slices:0 (1,)

Operations:  save/RestoreV2_25
save/RestoreV2_25 Input:  save/Const:0 ()
save/RestoreV2_25 Input:  save/RestoreV2_25/tensor_names:0 (1,)
save/RestoreV2_25 Input:  save/RestoreV2_25/shape_and_slices:0 (1,)
save/RestoreV2_25 Output:  save/RestoreV2_25:0 <unknown>

Operations:  save/Assign_25
save/Assign_25 Input:  conv5_3/filter:0 (3, 3, 512, 512)
save/Assign_25 Input:  save/RestoreV2_25:0 <unknown>
save/Assign_25 Output:  save/Assign_25:0 (3, 3, 512, 512)

Operations:  save/RestoreV2_26/tensor_names
save/RestoreV2_26/tensor_names Output:  save/RestoreV2_26/tensor_names:0 (1,)

Operations:  save/RestoreV2_26/shape_and_slices
save/RestoreV2_26/shape_and_slices Output:  save/RestoreV2_26/shape_and_slices:0 (1,)

Operations:  save/RestoreV2_26
save/RestoreV2_26 Input:  save/Const:0 ()
save/RestoreV2_26 Input:  save/RestoreV2_26/tensor_names:0 (1,)
save/RestoreV2_26 Input:  save/RestoreV2_26/shape_and_slices:0 (1,)
save/RestoreV2_26 Output:  save/RestoreV2_26:0 <unknown>

Operations:  save/Assign_26
save/Assign_26 Input:  fc6/biases:0 (4096,)
save/Assign_26 Input:  save/RestoreV2_26:0 <unknown>
save/Assign_26 Output:  save/Assign_26:0 (4096,)

Operations:  save/RestoreV2_27/tensor_names
save/RestoreV2_27/tensor_names Output:  save/RestoreV2_27/tensor_names:0 (1,)

Operations:  save/RestoreV2_27/shape_and_slices
save/RestoreV2_27/shape_and_slices Output:  save/RestoreV2_27/shape_and_slices:0 (1,)

Operations:  save/RestoreV2_27
save/RestoreV2_27 Input:  save/Const:0 ()
save/RestoreV2_27 Input:  save/RestoreV2_27/tensor_names:0 (1,)
save/RestoreV2_27 Input:  save/RestoreV2_27/shape_and_slices:0 (1,)
save/RestoreV2_27 Output:  save/RestoreV2_27:0 <unknown>

Operations:  save/Assign_27
save/Assign_27 Input:  fc6/weights:0 (7, 7, 512, 4096)
save/Assign_27 Input:  save/RestoreV2_27:0 <unknown>
save/Assign_27 Output:  save/Assign_27:0 (7, 7, 512, 4096)

Operations:  save/RestoreV2_28/tensor_names
save/RestoreV2_28/tensor_names Output:  save/RestoreV2_28/tensor_names:0 (1,)

Operations:  save/RestoreV2_28/shape_and_slices
save/RestoreV2_28/shape_and_slices Output:  save/RestoreV2_28/shape_and_slices:0 (1,)

Operations:  save/RestoreV2_28
save/RestoreV2_28 Input:  save/Const:0 ()
save/RestoreV2_28 Input:  save/RestoreV2_28/tensor_names:0 (1,)
save/RestoreV2_28 Input:  save/RestoreV2_28/shape_and_slices:0 (1,)
save/RestoreV2_28 Output:  save/RestoreV2_28:0 <unknown>

Operations:  save/Assign_28
save/Assign_28 Input:  fc7/biases:0 (4096,)
save/Assign_28 Input:  save/RestoreV2_28:0 <unknown>
save/Assign_28 Output:  save/Assign_28:0 (4096,)

Operations:  save/RestoreV2_29/tensor_names
save/RestoreV2_29/tensor_names Output:  save/RestoreV2_29/tensor_names:0 (1,)

Operations:  save/RestoreV2_29/shape_and_slices
save/RestoreV2_29/shape_and_slices Output:  save/RestoreV2_29/shape_and_slices:0 (1,)

Operations:  save/RestoreV2_29
save/RestoreV2_29 Input:  save/Const:0 ()
save/RestoreV2_29 Input:  save/RestoreV2_29/tensor_names:0 (1,)
save/RestoreV2_29 Input:  save/RestoreV2_29/shape_and_slices:0 (1,)
save/RestoreV2_29 Output:  save/RestoreV2_29:0 <unknown>

Operations:  save/Assign_29
save/Assign_29 Input:  fc7/weights:0 (1, 1, 4096, 4096)
save/Assign_29 Input:  save/RestoreV2_29:0 <unknown>
save/Assign_29 Output:  save/Assign_29:0 (1, 1, 4096, 4096)

Operations:  save/restore_shard

Operations:  save/restore_all
