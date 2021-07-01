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
