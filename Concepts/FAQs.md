  ## Concepts

  > ### What are channels and kernels?

  A channel is a collection of similar features.  
  A kernel is feature extractor, which extracts the occurrences of a particular feature from an image /input  into a channel.

  For example, let's say we want to extract horizontal edges from an image. A horizontal edge kernel will extract all the horizontal edges from the input and the resulting output after a convolution operation would be the channel which will contain the horizontal edges detected by the kernel.

  > ### Why should we (nearly) always use 3x3 kernels?

  3x3 kernels requires fewer operations to perform when convolving multiple layers to finally achieve global receptive field equal to image size. For example, let's assume that we have a 5x5 image. The convolution with a 5x5 kernel would require 25*25 = 125 multiplications to reach a layer with global receptive field of the entire image. 
  Suppose we use a 3x3 kernel, it would take 9x4 =36 multiplications for the first convolution and 1x9 =9 for the second convolution to get reach a layer with the same global receptive field.  This makes it much quicker for processing and saves on memory and time. 
  Another benefit of using a smaller filter is that finer details can be filtered out thereby increasing the chances of your network to make better predictions.  If you choose higher dimensional filters, information is lost during convolution.

  > ### How many times to we need to perform 3x3 convolutions operations to reach close to 1x1 from 199x199

  Let's type it out manually for fun

      199	x	199	=	197	x	197
      197	x	197	=	195	x	195
      195	x	195	=	193	x	193
      193	x	193	=	191	x	191
      191	x	191	=	189	x	189
      189	x	189	=	187	x	187
      187	x	187	=	185	x	185
      185	x	185	=	183	x	183
      183	x	183	=	181	x	181
      181	x	181	=	179	x	179
      179	x	179	=	177	x	177
      177	x	177	=	175	x	175
      175	x	175	=	173	x	173
      173	x	173	=	171	x	171
      171	x	171	=	169	x	169
      169	x	169	=	167	x	167
      167	x	167	=	165	x	165
      165	x	165	=	163	x	163
      163	x	163	=	161	x	161
      161	x	161	=	159	x	159
      159	x	159	=	157	x	157
      157	x	157	=	155	x	155
      155	x	155	=	153	x	153
      153	x	153	=	151	x	151
      151	x	151	=	149	x	149
      149	x	149	=	147	x	147
      147	x	147	=	145	x	145
      145	x	145	=	143	x	143
      143	x	143	=	141	x	141
      141	x	141	=	139	x	139
      139	x	139	=	137	x	137
      137	x	137	=	135	x	135
      135	x	135	=	133	x	133
      133	x	133	=	131	x	131
      131	x	131	=	129	x	129
      129	x	129	=	127	x	127
      127	x	127	=	125	x	125
      125	x	125	=	123	x	123
      123	x	123	=	121	x	121
      121	x	121	=	119	x	119
      119	x	119	=	117	x	117
      117	x	117	=	115	x	115
      115	x	115	=	113	x	113
      113	x	113	=	111	x	111
      111	x	111	=	109	x	109
      109	x	109	=	107	x	107
      107	x	107	=	105	x	105
      105	x	105	=	103	x	103
      103	x	103	=	101	x	101
      101	x	101	=	99	x	99
      99	x	99	=	97	x	97
      97	x	97	=	95	x	95
      95	x	95	=	93	x	93
      93	x	93	=	91	x	91
      91	x	91	=	89	x	89
      89	x	89	=	87	x	87
      87	x	87	=	85	x	85
      85	x	85	=	83	x	83
      83	x	83	=	81	x	81
      81	x	81	=	79	x	79
      79	x	79	=	77	x	77
      77	x	77	=	75	x	75
      75	x	75	=	73	x	73
      73	x	73	=	71	x	71
      71	x	71	=	69	x	69
      69	x	69	=	67	x	67
      67	x	67	=	65	x	65
      65	x	65	=	63	x	63
      63	x	63	=	61	x	61
      61	x	61	=	59	x	59
      59	x	59	=	57	x	57
      57	x	57	=	55	x	55
      55	x	55	=	53	x	53
      53	x	53	=	51	x	51
      51	x	51	=	49	x	49
      49	x	49	=	47	x	47
      47	x	47	=	45	x	45
      45	x	45	=	43	x	43
      43	x	43	=	41	x	41
      41	x	41	=	39	x	39
      39	x	39	=	37	x	37
      37	x	37	=	35	x	35
      35	x	35	=	33	x	33
      33	x	33	=	31	x	31
      31	x	31	=	29	x	29
      29	x	29	=	27	x	27
      27	x	27	=	25	x	25
      25	x	25	=	23	x	23
      23	x	23	=	21	x	21
      21	x	21	=	19	x	19
      19	x	19	=	17	x	17
      17	x	17	=	15	x	15
      15	x	15	=	13	x	13
      13	x	13	=	11	x	11
      11	x	11	=	9	x	9
      9	x	9	=	7	x	7
      7	x	7	=	5	x	5
      5	x	5	=	3	x	3
      3	x	3	=	1	x	1

  So in total it would take around 99 convolutions

  > ### How are kernels initialized?

  Kernels are intialized with random numbers between 0 and 1. Having a lower and upper limit allows the neural net to understand relative magnitude and make more appropriate changes to the values of the kernel. 

  > ### What happens during the training of a DNN?
  > 
  During training, the DNN extracts features from the input image and combines them to form textures. Many such textures are combined to form patterns which turn are combined to form parts of objects. Parts of objects and combined to form the objects themselves which are finally compared to the label of the training data. 

  When kernels convolve over images, they produce channels  which are nothing but collection of similar features. Depending on how many layers you choose, each layer will become an aggregation of channels of the previous layer. Ultimately, it will trickle down to the final layer or output layer who's values will be used to make predictions.

  During training, these output values are compared to the expected values and the magnitude and direction of the deviation is captured and fed back to the network. The network then makes small incremental changes backwards (like reverse engineering) to figure out what would be the optimum values of the kernels to arrive at the expected output. These small changes are going to be accrued over many such training images until we get the degree of accuracy we need. 

  Think of it like practicing a musical instrument. When you want to play a song, you'll first have a recording of it. You'll play it out based on your memory and your mentor will first correct any basic flaws. Then you play it again and make finer changes until you're fairly good at it. The more number of songs you practice in this way, the better you will get at identifying and playing new songs accurately. 



