# Text-to-image-search
This project is based on image search based on text. For a given text (query), the main function will return top n images relevant to the query.

Main Function-
```python

def searched_images(params):
    image_features, image_attributes = get_image_features(params.get("directory"))
    text_features = encode_search_query(params.get("search_query"))
    best_photo_ids, similarity = find_best_matches(text_features, image_features.cpu().numpy(), image_attributes,
                                                   params.get("N_images", 3))
    list_res = []
    for i in range(len(best_photo_ids)):
        dict_ = {"Photo_id": best_photo_ids[i], "position": i + 1, "score": similarity[i]}
        list_res.append(dict_)
    for i in best_photo_ids:
        img = cv2.imread(
            os.path.join(params.get("directory"), i))
        resized_image = cv2.resize(img, (1000, 700))
        cv2.imshow("sample1", resized_image)
        cv2.waitKey(0)
    return list_res
    
```
Main function takes input as params which is a dictionary containing directory of images, Search query and No of top images we want to search.
Input Format-
```python
params = {"directory": <path of directory of images>,
              "search_query": <query to be searched> # for ex - "Children playing footall",
              "N_images": <number of top images which are relevant to the query> # integer}
```
Sample Input format-
```python
    params = {"directory": "natural-language-joint-query-search\images",
              "search_query": "Children playing football",
              "N_images": 3}
```

Sample Output format-
Output contains a list of dictionaries shown below
```python
[{'Photo_id': 'child-613199_1280.jpg', 'position': 1, 'score': 0.28278786}, {'Photo_id': 'children-1822704_1920.jpg', 'position': 2, 'score': 0.24237464}, {'Photo_id': 'water-863053_1920.jpg', 'position': 3, 'score': 0.21144849}]
```
