# Credit Card OCR Tesseract Python
This is a hobby project  to develop a tool to read credit cards with Tesseract and simple operations of computer vision. 
`Tesseract` is an optical character recognition engine for various operating systems. It works great and could be trained on a custom dataset. In the case of credit card recognition, we could train `Tesseract` on **OCR-A** font to have a better result. But in this project, I just wanna show the image processing to localize 16 digits and other information such as holder card name, expiration date.... 
### Filtering image
We can start with a mastercard photo, convert it to grayscale and fix its width to 666px
![gray](/img/gray.png)

Then applying a `Top-Hat filter` to enhance bright objects of interest in a dark background.  It is useful in observing the minor details of the inputs when are present as light pixels on a dark background. 
```python
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT,sqKernel)
```
![gray](/img/tophat.png)

Now we perform a binary and Otsu threshold to have a binary image
```python
thresh = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
```
![gray](/img/thresh.png)

At this step, most of the noise has been filtered, but we need a few more steps to detect the location of relevant informations (digits , texts).
We perform a `closing` transformation to detect lines of characters and a `opening` transformation to reduce noises. 
```python
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rectKernel)
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, sqKernel)
```
![gray](/img/closing.png)
![gray](/img/opening.png)

### Characters Localization 
After filtering image, we can find contours to detect groups of digits and texts.
```python
cnts = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
```
![gray](/img/contours.png)

Deleting small contours, and find rectangle ones.
Now, we can use `Tesseract` to recognize characters in regions of interest that we found.
![gray](/img/tesseract.png)
