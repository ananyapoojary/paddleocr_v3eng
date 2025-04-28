const sendImageToApi = async () => {
  if (!imageUri) {
    Alert.alert('No image', 'Pick an image first!');
    return;
  }

  const formData = new FormData();
  formData.append('file', {
    uri: imageUri,
    name: 'photo.jpg',
    type: 'image/jpeg',
  });

  try {
    const res = await axios.post(
      'https://your-real-render-url.onrender.com/ocr',
      formData,
      { headers: { 'Content-Type': 'multipart/form-data' } }
    );
    setOcrResult(res.data.results); // .results not .data
  } catch (err) {
    console.error(err);
    Alert.alert('Error', 'Failed to process image');
  }
};
