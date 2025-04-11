import React, { useState } from 'react';
import { DropzoneArea } from 'material-ui-dropzone';
import axios from 'axios';
import "../index.css";

const DragAndDrop = () => {
  const [data, setData] = useState(null);
  const [files, setFiles] = useState([]);
  const [image, setImage] = useState(false);
  const [predicted, setPredicted] = useState(false); 

  const handleFileChange = (newFiles) => {
    setFiles(newFiles);
    setImage(true);
    setPredicted(false); 
    console.log('Files:', newFiles);
  };

  const handleDeleteImage = () => {
    setFiles([]);
    setImage(false);
    setData(null); 
    setPredicted(false); 
  };

  const handlePredict = async () => {
    await sendFile(); 
    setPredicted(true); 
  };

  const sendFile = async () => {
    if (image && files.length > 0) {
      let formData = new FormData();
      formData.append("file", files[0]); // Use files[0] to append the first file
      try {
        let res = await axios.post("http://localhost:8000/predict", formData);
        if (res.status === 200) {
          console.log("predicted:", res.data);
          setData(res.data); 
        }
      } catch (error) {
        console.error("Error uploading file:", error);
      }
    }
  };

  return (
    <div className="drag-and-drop">
      <h2>{files.length > 0 ? "Image Uploaded Successfully" : "Upload Plant Leaf Images"}</h2>

      {files.length === 0 ? (
        <DropzoneArea
          acceptedFiles={['image/jpeg', 'image/png', 'image/bmp']}
          dropzoneText={"Drag and drop an image here or click"}
          onChange={handleFileChange}
          filesLimit={1}
          maxFileSize={5000000}
          showAlerts={true}
        />
      ) : (
        <div className="image-preview">
          <img src={URL.createObjectURL(files[0])} alt="Uploaded" />
          <button onClick={handleDeleteImage} className="delete-button">Delete</button>
          {!predicted && (
            <button onClick={handlePredict} className="delete-button">Predict</button>
          )}
        </div>
      )}

      {data && (
        <div className="prediction-result">
          <h3>Prediction Result:</h3>
          <p>Class: {data.class}</p>
          <p>Confidence: {data.confidence}%</p>
        </div>
      )}
    </div>
  );
};

export default DragAndDrop;
