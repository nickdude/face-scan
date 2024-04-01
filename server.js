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

  if ( !getDirNames().includes(socket.id)){
    const directoryPath = `./${socket.id}`;
    fs.mkdir(directoryPath, { recursive: true }, (err) => {
        if (err) {
          // Handle error if directory creation fails
          console.error("Error creating directory:", err);
        } else {
          // Directory created successfully
          console.log(`Client session directory created successfully => ${socket.id}`);
        }
      });

      let textData = "";


  socket.on("message", (data) => {
    textData += data + "\n"; // Append the received string and a newline character
    stringCounter++;

    const fileName = `user_${socket.id}_text.txt`;
    const filePath = path.join(directoryPath, fileName);

    fs.writeFile(filePath, textData, (err) => {
      if (err) {
        console.error("Error writing text data:", err);
        return 500;
      }


      console.log(
        `Text data written to ${filePath} => Iteration ${stringCounter}`
      );
      

      if (stringCounter === 900) {

        console.log("Executing python code")
        // // Execute Python script
        // const pythonScript = `python practice.py ${textDir} ${fileName}`;
        // exec(pythonScript, (error, stdout, stderr) => {
        //   if (error) {
        //     console.error(`Error executing Python script: ${error}`);
        //     return;
        //   }

        //   console.log(`Python output: ${stdout}`);

        //   // Read and emit JSON data
        //   const jsonFilePath = path.join(
        //     textDir,
        //     fileName.replace(".txt", ".json")
        //   );
        //   fs.readFile(jsonFilePath, "utf8", (err, jsonData) => {
        //     if (err) {
        //       console.error("Error reading JSON data:", err);
        //       return;
        //     }

        //     socket.emit("json", jsonData);
        //   });
        // });

        // stringCounter = 0; // Reset the string counter after processing
      }
    });

    socket.emit("message_received")

  });

  socket.on("disconnect", () => {
    console.log(`User disconnected: ${socket.id}`);

    if ( getDirNames().includes(socket.id)){
        fs.rmdir(`./${socket.id}`, (err)=>{
            if (err) {
                console.log(`Error removing directory => ${socket.id}`)
                return
            }
        })
        return
    }

    stringCounter = 0; // Reset the string counter on disconnect
  });
  }
  else{
    // TODO: handle by ending the session completely
    return 500; // temp solution for now
  }

});


port = 5000
server.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});
