// const express = require('express');
// const http = require('http');
// const socketIO = require('socket.io');
// const { exec } = require('child_process');
// const fs = require('fs');
// const path = require('path');

// const app = express();
// const server = http.createServer(app);
// const io = socketIO(server);

// const base64Dir = './base64_data/';
// const jsonDir = './json_data/';

// // Create directories if they don't exist
// if (!fs.existsSync(base64Dir)) {
//     fs.mkdirSync(base64Dir);
// }

// if (!fs.existsSync(jsonDir)) {
//     fs.mkdirSync(jsonDir);
// }

// // Serve the index.html file
// app.get('/', (req, res) => {
//     res.sendFile(path.join(__dirname, 'index.html'));
// });

// io.on('connection', (socket) => {
//     console.log(`User connected: ${socket.id}`);

//     let base64Data = '';

//     socket.on('stream', (data) => {
//         base64Data += data;
//     });

//     socket.on('streamEnd', () => {
//         const fileName = `user_${socket.id}.txt`;
//         const filePath = path.join(base64Dir, fileName);

//         fs.writeFile(filePath, base64Data, 'base64', (err) => {
//             if (err) {
//                 console.error('Error writing base64 data:', err);
//                 return;
//             }

//             console.log(`Base64 data written to ${filePath}`);

//             // Execute Python script
//             const pythonScript = `python practice.py ${base64Dir} ${jsonDir} ${fileName}`;
//             exec(pythonScript, (error, stdout, stderr) => {
//                 if (error) {
//                     console.error(`Error executing Python script: ${error}`);
//                     return;
//                 }

//                 console.log(`Python output: ${stdout}`);

//                 // Read and emit JSON data
//                 const jsonFilePath = path.join(jsonDir, fileName.replace('.txt', '.json'));
//                 fs.readFile(jsonFilePath, 'utf8', (err, jsonData) => {
//                     if (err) {
//                         console.error('Error reading JSON data:', err);
//                         return;
//                     }

//                     socket.emit('json', jsonData);
//                 });
//             });
//         });
//     });

//     socket.on('disconnect', () => {
//         console.log(`User disconnected: ${socket.id}`);
//     });
// });

// server.listen(3000, () => {
//     console.log('Server running on http://localhost:3000');
// });

const express = require("express");
const http = require("http");
const socketIO = require("socket.io");
const { exec } = require("child_process");
const fs = require("fs");
const path = require("path");

const app = express();
const server = http.createServer(app);
const io = socketIO(server);

// utility functions

const separationKeyword = "faceScanImageSeparationIdentifier";

const getDirNames = () => {
  var dirNames = [];
  const rootDir = "./";
  fs.readdir(rootDir, (err, files) => {
    if (err) {
      console.error("Error reading directory:", err);
      return;
    }

    files.forEach((file) => {
      const filePath = path.join(rootDir, file);
      fs.stat(filePath, (err, stats) => {
        if (err) {
          console.error("Error getting file stats:", err);
          return;
        }
        if (stats.isDirectory()) {
          dirNames.push(file.toString());
        }
      });
    });
  });

  return dirNames;
};

let stringCounter = 0;

// Create directory if it doesn't exist
// if (!fs.existsSync(textDir)) {
//   fs.mkdirSync(textDir);
// }

// Serve the index.html file
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "index.html"));
});

io.on("connection", (socket) => {
  console.log(`User connected: ${socket.id}`);

  // if no client directory, make one

  if (!getDirNames().includes(socket.id)) {
    const directoryPath = `./${socket.id}`;
    fs.mkdir(directoryPath, { recursive: true }, (err) => {
      if (err) {
        // Handle error if directory creation fails
        console.error("Error creating directory:", err);
      } else {
        // Directory created successfully
        console.log(
          `Client session directory created successfully => ${socket.id}`
        );
      }
    });

    let textData = "";

    socket.on("message", (data) => {
      textData = "";
      textData += data + "\n"; // Append the received string and a newline character
      stringCounter++;

      var index = textData.indexOf(separationKeyword);

      var frameNumber = textData.slice(0, index);

      const fileName = `user_${socket.id}_text_${frameNumber}.txt`;
      const filePath = path.join(directoryPath, fileName);

      fs.writeFile(filePath, textData, (err) => {
        if (err) {
          console.error("Error writing text data:", err);
          return 500;
        }

        console.log(
          `Text data written to ${filePath} => Iteration ${stringCounter}`
        );

        fs.readFile(filePath, "utf8", (err, data) => {
          if (err) {
            // Handle error if reading file fails
            console.error("Error reading file:", err);
            return;
          }

          console.log(
            `slicing after  ${
              index + separationKeyword.length
            } for frame ${stringCounter}`
          );
          const base64String = data.slice(index + separationKeyword.length, data.length - 1);

          const imageBuffer = Buffer.from(base64String, "base64");

          fs.writeFile(`./${socket.id}/output${frameNumber}.png`, imageBuffer, (err) => {
            if (err) throw err;
            console.log("The file has been saved!");
          });
        });

        // fs.rm(filePath, (err)=>{
        //     if ( err) {
        //         "Error deleting png converted file"   
        //     }
        // })

        if (stringCounter === 800) {
          console.log("Executing python code");

          // Execute Python script
          const pythonScript = `python3 practice.py ${socket.id}`;
          exec(pythonScript, (error, stdout, stderr) => {
            if (error) {
              console.error(`Error executing Python script: ${error}`);
              return;
            }


            console.log("Inside executing python script")

            if ( stderr) {
                console.error("Python err", stderr)
            }
            console.log(`Python output: ${stdout}`);

            // Read and emit JSON data
            const jsonFilePath = path.join(
              directoryPath,
              `${socket.id}_results.json`
            );
            fs.readFile(jsonFilePath, "utf8", (err, jsonData) => {
              if (err) {
                console.error("Error reading JSON data:", err);
                return;
              }
              const parsedData = JSON.parse(jsonData); // Parse the string to a JSON object

              console.log(parsedData)

              socket.emit("results", parsedData);
            });
          });

          stringCounter = 0; // Reset the string counter after processing
        }
      });

    });

    socket.on("disconnect", () => {
      console.log(`User disconnected: ${socket.id}`);

      if (getDirNames().includes(socket.id)) {
        fs.rmdir(`./${socket.id}`, (err) => {
          if (err) {
            console.log(`Error removing directory => ${socket.id}`);
            return;
          }
        });
        return;
      }

      stringCounter = 0; // Reset the string counter on disconnect
    });
  } else {
    // TODO: handle by ending the session completely
    return 500; // temp solution for now
  }
});

port = 5000;
server.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});
