# OPPLY
Opply Coding Task - Image Similarity 


After reading the task description thoroughly I understood that the key objective is image similarity, precisely image similarity to the keyword - “almonds”. From my overall experience in the Machine Learning field of Image recognition, image classification, image similarity etc. I recalled the world known OpenAI CLIP model. 

“CLIP is a neural network trained on a variety of (image,text) paurs” (source : https://huggingface.co/openai/clip-vit-base-patch32 ). 

The main idea is to encode a text into an embedded vector (in this case 1x512), as well as encode an image into an embedded vector of 1x512, and use a measure (in this case cosine-similarity) to find the most appropriate result of a text for a given image. 

“It is trained on 400,000,000 (image, text) pairs. An (image, text) pair might be a picture and its caption. So this means that there are 400,000,000 pictures and their captions that are matched up, and this is the data that is used in training the CLIP model.” (source : https://www.kdnuggets.com/2021/03/beginners-guide-clip-model.html ) 

From the list of pretrained CLIP models : https://huggingface.co/models?filter=clip 

I used the “clip-vit-base-patch32” model, as it was commonly used in the examples of implementations of the CLIP models, and therefore I decided to also use it in my solution for this coding task. The “clip-vit-base-patch32” was also used in the documentations 

here: 
https://github.com/openai/CLIP 
or here: 
https://huggingface.co/transformers/v4.6.0/model_doc/clip.html 

Code : 

After importing the necessary libraries, I began writing the CLIPFeatureExtractor class. The two main  sources for this code were the hugging face documentation on CLIP FeatureExtractor and the Github openai/CLIP source code: 

The class consisted of a constructor and 2 functions, namely “encode_text” and “encode_images” . After loading the pretrained model for both the model and the processor, both the text and image features were normalized and both functions return the vector embedded vector of text_features and image_featues. 

Next, after creating an instance of the CLIPFeatureExtractor class, the encode_text function was used to create an embedded vector of the key word “almonds”.  The load_image() function is used for opening images and converting them into the RGB colour scale. 


The make_predictions_for_directory() function will be a function that takes all images from the relevant directory path and after using load_image, it uses the encode_images functions to return the image_features vector for all the images from the given directory. Next, the “similarities” are calculated using the Cosine Similarity function, comparing the images embedded vector values to the embedded text values of the word “almonds”. The function returns a mean of all the cosine similarity values from the given directory path. 

Now two seperate function were created 

make_preds_for_examples() takes in the directory path and joins the path for both Relevant and Irrelevant subfolders. Then we iterate through all the elements inside the Relevant and Irrelevant directories, and create a prediction vector, which uses the first made_predictions_for_directory() function. This particular function returns two dictionaries, one with the predictions for each of the directories, and one with the scaled predictions of each of the directories using the MinMax scaler.

The second function make_preds_for_google_images() goes through very similar procedures as the first function, with slightly altered directory path list creation etc, and ultimately returning only one dictionary, which is the predictions for each of the 10 categories of products. 

RESULTS

The higher the cosine similarity the more “goodness” it has. We can observe there is a significantly higher score for the Relevant Examples than for Irrelevant examples. For the examples of application of the CLIP models in image similarity, some examples have shown the results of around 25-30% to be in fact a significant result. 

For the other dataset - the google_images, the results were also highest for the “almonds” category, with “hazelnut” category running at close second, however the main results for objective interpretation of the outcomes are the Example datasets for Relevant and Irrelevant images.  

This model is flexible and can easily be implemented for a different word than “almonds”, as it is a matter of one line of code to create a different embedded word vector and proceed with creating new dictionaries of predictions for the given word. 
 
Overall, using a pre-trained model for this particular coding task with limited time and resources was ultimately the most efficient way of completing the main objectives. 

What Could be improved given more resources and time, and other reflections

I have thought of a number of adjustments and alterations that could potentially be implemented in this particular application of the CLIP model. 

The first thing that came to my mind is to make use of the google_images dataset. From my understanding, that dataset was given for any potential training model purposes, and the test set was meant to be the Examples of Relevant and Irrelevant classes. I did predictions on the google_images dataset just for experimental purposes, however I believe that a better use of that dataset would be to “tune” the CLIP model by training it on the additional 500 images with embedded text labels, however this process would require more of my time spent on research and coding. 

Next, I believe there there could be a simplified one general function for predictions, instead of implementing two seperate functions. I also had to remove a few webp files, which in retrospect could have been autoamtically done by implementing a line of code looking something like this : 

“(...) if filename.endswith(“png”) or filename.endswith(“jpg”) : then … 

I believe the problem was related to pictures in the RGBA format, which I would have dived into and try to overcome if I had more time, however removing 4 images from this dataset did not really cause any much damage or loss of information. 

Another thing I thought about is perhaps using a different distance function, such as the Euclidean distance instead of the Cosine Similarity. Also, there is a number of CLIP models and they if I had more time I would have tested a variety of models out, to compare the results ultimately finding the most efficient of the models. I have found a few articles regarding tuning the model, adding parameters etc, which is another thing that would be considered for improving the efficiency of the model, given more time and resources. 






Sources:

https://huggingface.co/transformers/v4.6.0/model_doc/clip.html#transformers.CLIPFeatureExtractor 

https://github.com/openai/CLIP

https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/feature_extraction_clip.py 

https://www.kdnuggets.com/2021/03/beginners-guide-clip-model.html

https://www.reddit.com/r/computervision/comments/mm2mz6/simple_implementation_of_openai_clip_model_in/

https://towardsdatascience.com/how-to-train-your-clip-45a451dcd303 
