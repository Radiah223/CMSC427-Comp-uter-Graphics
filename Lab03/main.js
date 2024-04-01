/** @type {HTMLCanvasElement} */
const canvas = document.querySelector('#c');
//Getting access to WebGL 
/** @type {WebGLRenderingContext} */
const gl = canvas.getContext("webgl2"); 
if(!gl){ 
}

"use strict"; 
const simpleVertexShader = `#version 300 es
in vec4 a_position;
in vec2 a_texcoord;
in vec3 a_normal;

uniform mat4 u_matrix;

out vec2 v_texcoord;
out vec3 v_normal;

void main() {
  // Multiply the position by the matrix.
  gl_Position = u_matrix * a_position;

  // Pass the texcoord to the fragment shader.
  v_texcoord = a_texcoord;
  v_normal = a_normal;
}
`;

const specularVertexShader = `#version 300 es
in vec4 a_position;
in vec3 a_normal;

uniform vec3 u_lightWorldPosition;
uniform vec3 u_viewWorldPosition;

uniform mat4 u_world;
uniform mat4 u_worldViewProjection;
uniform mat4 u_worldInverseTranspose;

out vec3 v_normal;
out vec3 v_surfaceToLight;
out vec3 v_surfaceToView;

void main() {
  // Multiply the position by the matrix.
  gl_Position = u_worldViewProjection * a_position;

  // orient the normals and pass to the fragment shader
  v_normal = mat3(u_worldInverseTranspose) * a_normal;

  // compute the world position of the surfoace
  vec3 surfaceWorldPosition = (u_world * a_position).xyz;

  // compute the vector of the surface to the light
  // and pass it to the fragment shader
  v_surfaceToLight = u_lightWorldPosition - surfaceWorldPosition;

  // compute the vector of the surface to the view/camera
  // and pass it to the fragment shader
  v_surfaceToView = u_viewWorldPosition - surfaceWorldPosition;
}
`;

const normalShader = `#version 300 es
precision highp float;

// Passed in from the vertex shader.
in vec2 v_texcoord;
in vec3 v_normal;

out vec4 outColor;

void main() {
  outColor = vec4(v_normal * .5 + .5, 1);
  
}
`;

const textureShader = `#version 300 es
precision highp float;

// Passed in from the vertex shader.
in vec2 v_texcoord;
in vec3 v_normal;

uniform sampler2D u_texture;

out vec4 outColor;

void main() {
  outColor = texture(u_texture, v_texcoord);
}
`;

const specularShader = `#version 300 es
#if GL_FRAGMENT_PRECISION_HIGH
  precision highp float;
#else
  precision highp float;
#endif

// Passed in from the vertex shader.
in vec3 v_normal;
in vec3 v_surfaceToLight;
in vec3 v_surfaceToView;

uniform vec4 u_color;
uniform float u_shininess;

out vec4 outColor;

void main() {
  // because v_normal is a varying it's interpolated
  // so it will not be a unit vector. Normalizing it
  // will make it a unit vector again
  vec3 normal = normalize(v_normal);

  vec3 surfaceToLightDirection = normalize(v_surfaceToLight);
  vec3 surfaceToViewDirection = normalize(v_surfaceToView);
  vec3 halfVector = normalize(surfaceToLightDirection + surfaceToViewDirection);

  float light = dot(normal, surfaceToLightDirection);
  float specular = 0.0;
  if (light > 0.0) {
    specular = pow(dot(normal, halfVector), u_shininess);
  }

  outColor = u_color;

  // Lets multiply just the color portion (not the alpha)
  // by the light
  outColor.rgb *= light;

  // Just add in the specular
  outColor.rgb += specular;
}
`;

const spotlightVertex = `#version 300 es
in vec4 a_position;
in vec3 a_normal;

uniform vec3 u_lightWorldPosition;
uniform vec3 u_viewWorldPosition;

uniform mat4 u_world;
uniform mat4 u_worldViewProjection;
uniform mat4 u_worldInverseTranspose;

out vec3 v_normal;

out vec3 v_surfaceToLight;
out vec3 v_surfaceToView;

void main() {
  // Multiply the position by the matrix.
  gl_Position = u_worldViewProjection * a_position;

  // orient the normals and pass to the fragment shader
  v_normal = mat3(u_worldInverseTranspose) * a_normal;

  // compute the world position of the surfoace
  vec3 surfaceWorldPosition = (u_world * a_position).xyz;

  // compute the vector of the surface to the light
  // and pass it to the fragment shader
  v_surfaceToLight = u_lightWorldPosition - surfaceWorldPosition;

  // compute the vector of the surface to the view/camera
  // and pass it to the fragment shader
  v_surfaceToView = u_viewWorldPosition - surfaceWorldPosition;
}
`;

const spotlightShader = `#version 300 es
#if GL_FRAGMENT_PRECISION_HIGH
  precision highp float;
#else
  precision highp float;
#endif

// Passed in from the vertex shader.
in vec3 v_normal;
in vec3 v_surfaceToLight;
in vec3 v_surfaceToView;

uniform vec4 u_color;
uniform float u_shininess;
uniform vec3 u_lightDirection;
uniform float u_limit;          // in dot space

// we need to declare an output for the fragment shader
out vec4 outColor;

void main() {
  // because v_normal is a varying it's interpolated
  // so it will not be a unit vector. Normalizing it
  // will make it a unit vector again
  vec3 normal = normalize(v_normal);

  vec3 surfaceToLightDirection = normalize(v_surfaceToLight);
  vec3 surfaceToViewDirection = normalize(v_surfaceToView);
  vec3 halfVector = normalize(surfaceToLightDirection + surfaceToViewDirection);

  float light = 0.0;
  float specular = 0.0;
  float dotFromDirection = dot(surfaceToLightDirection,
                               -u_lightDirection);
  if (dotFromDirection >= u_limit) {
    light = dot(normal, surfaceToLightDirection);
    if (light > 0.0) {
      specular = pow(dot(normal, halfVector), u_shininess);
    }
  }

  outColor = u_color;

  // Lets multiply just the color portion (not the alpha)
  // by the light
  outColor.rgb *= light;

  // Just add in the specular
  outColor.rgb += specular;
}
`;


function main() {


  twgl.setDefaults({
    attribPrefix: "a_"
  });

  // Get A WebGL context
  /** @type {HTMLCanvasElement} */
  const canvas = document.querySelector("canvas");
  const gl = canvas.getContext("webgl2");
  if (!gl) {
    return;
  }

  const data = {
    tolerance: 0.15,
    distance: 1,
    divisions: 16,
    startAngle: 0,
    endAngle: Math.PI * 2,
    capStart: true,
    capEnd: true,
    triangles: true,
    maxAngle: degToRad(30),
    mode: 0,
    shape: 0,
    orthographic: false,
    light: 0
  };
  
function makeTetra() {
const positions = [0.5,0.5,0.5, -0.5,0.5,0.5, 0.5,0.5,-0.5, 0.5,-0.5,0.5];
const texcoords = [0,0,   0,0,   0,0,   0,0];
const indices =  [ 0,2,1, 0,1,3, 1,2,3, 0,3,2 ];
    return {
      position: positions,
      texcoord: texcoords,
      indices: indices,
    };
  }

  function generateMesh(bufferInfo) {

    const points2 = [ // Top to bottom
      [100, 0],
      [50,100],
      [60,150],
      [50, 200]
    ];
   //const tempArrays = lathePoints(points2, data.startAngle, data.endAngle, data.divisions, data.capStart, data.capEnd);
   var tempArrays;
   if (data.shape == 0) {
      tempArrays = makeTetra();
      }
   else {
      tempArrays = lathePoints(points2, data.startAngle, data.endAngle, data.divisions, data.capStart, data.capEnd);
   
   }
   
    const arrays = generateNormals(tempArrays, data.maxAngle);
    const extents = getExtents(arrays.position);
    if (!bufferInfo) {
      // calls gl.createBuffer, gl.bindBuffer, and gl.bufferData for each array
      bufferInfo = twgl.createBufferInfoFromArrays(gl, arrays);
      // calls gl.createVertexArray, gl.bindVertexArray,
      // and then gl.bindBuffer, gl.enableVertexAttribArray, gl.vertexAttribPointer for each attribute
      vao = twgl.createVAOFromBufferInfo(gl, programInfos[0], bufferInfo);
    } else {
      gl.bindBuffer(gl.ARRAY_BUFFER, bufferInfo.attribs.a_position.buffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(arrays.position), gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, bufferInfo.attribs.a_texcoord.buffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(arrays.texcoord), gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, bufferInfo.attribs.a_normal.buffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(arrays.normal), gl.STATIC_DRAW);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, bufferInfo.indices);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(arrays.indices), gl.STATIC_DRAW);
      bufferInfo.numElements = arrays.indices.length;
    }
    return {
      bufferInfo: bufferInfo,
      extents: extents,
    };
  }

  // used to force the locations of attributes so they are
  // the same across programs
  const attributes = ['a_position', 'a_texcoord', 'a_normal'];
  // setup GLSL programs
  const programInfos = [
    // compiles shaders, links program and looks up locations
    twgl.createProgramInfo(gl, [simpleVertexShader, normalShader], attributes),
    twgl.createProgramInfo(gl, [specularVertexShader, specularShader], attributes),
    twgl.createProgramInfo(gl, [simpleVertexShader, textureShader], attributes),
    twgl.createProgramInfo(gl,[spotlightVertex,spotlightShader],attributes)
  ];

  const texInfo = loadImageAndCreateTextureInfo("https://webgl2fundamentals.org/webgl/resources/uv-grid.png", render);
  let worldMatrix = m4.identity();
  //adding a temporary translation matrix
  let tempMatrix = m4.identity();
  let projectionMatrix;
  let extents;
  let bufferInfo;
  let vao;
  

  function update() {
    const info = generateMesh(bufferInfo);
    extents = info.extents;
    bufferInfo = info.bufferInfo;
     
    tempMatrix = m4.identity();
    //worldMatrix = m4.identity();
    const centroid = [(extents.min[0] + extents.max[0]) / 4, (extents.min[1] + extents.max[1]) / 4, (extents.min[2] + extents.max[2]) / 4];
    
    tempMatrix = m4.translate(worldMatrix,-centroid[0], -centroid[1], -centroid[2]);
    //adding translation to worldMatrix
    worldMatrix = m4.multiply(tempMatrix, m4.translation(centroid[0], centroid[1], centroid[2])); 
    
    console.log(data.shape);
    render();
  }
  update();

  function render() {
  
  console.log(data.shape);
    twgl.resizeCanvasToDisplaySize(gl.canvas, window.devicePixelRatio);

    // Tell WebGL how to convert from clip space to pixels
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

    gl.enable(gl.DEPTH_TEST);

    // Clear the canvas AND the depth buffer.
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);


    // Compute the projection matrix
    const fieldOfViewRadians = Math.PI * .25;
    const aspect = gl.canvas.clientWidth / gl.canvas.clientHeight;

    //adjusting for orthographic view 
    projectionMatrix = m4.perspective(fieldOfViewRadians, aspect, 0.5, -0.5); 


    if (data.orthographic){
      if(data.shape == 0){ 
        projectionMatrix = m4.orthographic(-2*data.distance,2*data.distance,-2*data.distance,2*data.distance,1,2000); 
      }else{
        projectionMatrix = m4.orthographic(-data.distance*gl.canvas.clientWidth/2,data.distance*gl.canvas.clientWidth/2,-data.distance*gl.canvas.clientHeight/2,data.distance*gl.canvas.clientHeight/2,1,2000); 
      }
    }

    // Compute the camera's matrix using look at.
    const midY = lerp(extents.min[1], extents.max[1], .5);
    const sizeToFitOnScreen = (extents.max[1] - extents.min[1]) *5; // Scaling!
    const distance = data.distance*sizeToFitOnScreen;
    const cameraPosition = [0, midY, distance];
    const target = [0, midY, 0];
    const up = [0, -1, 0]; // we used 2d points as input which means orientation is flipped
    const cameraMatrix = m4.lookAt(cameraPosition, target, up);

    // Make a view matrix from the camera matrix.
    const viewMatrix = m4.inverse(cameraMatrix);

    const viewProjectionMatrix = m4.multiply(projectionMatrix, viewMatrix);

    const programInfo = programInfos[data.mode];
    gl.useProgram(programInfo.program);

    // Setup all the needed attributes.
    gl.bindVertexArray(vao);

    const worldViewProjection = m4.multiply(viewProjectionMatrix, worldMatrix);

    let lightPos = [midY * 1.5, midY * 2, distance * 1.5];
    console.log(lightPos); 
    if(data.light ==1){ 
      if (data.shape == 0){ 
        lightPos = [1,1,1]
      }else{ 
        lightPos = [450,500,1200]
      }
    }else if(data.light == 2){ 
      if(data.shape == 0){ 
        lightPos = [-1,-1,1]

      }else{ 
        lightPos = [500,500,1000]
      }

    }else if (data.light == 3){ 
      if (data.shape == 0){ 
        lightPos =[-1,1,0]
      }else{ 
        lightPos = [500,500,500]
      }
    }else if (data.light == 4){ 
      if(data.shape == 0){ 
        lightPos =[0.3,0.3,1]; 
        direction = [0,0,-1]; 
      }else{ 
        lightPos = [40,70,120];
        direction = [-0.310,-0.194,-0.93];
      } 
    }


    // Set the uniforms
    // calls gl.uniformXXX, gl.activeTexture, gl.bindTextureCS718PolyhedralMeshes
    if(data.light == 4){
      twgl.setUniforms(programInfo, {
      u_matrix: worldViewProjection,
      u_worldViewProjection: worldViewProjection,
      u_world: worldMatrix,
      u_worldInverseTranspose: m4.transpose(m4.inverse(worldMatrix)),
      u_lightWorldPosition: lightPos,
      u_lightDirection:direction, 
      u_limit: Math.cos(degToRad(25)),
      u_viewWorldPosition: cameraMatrix.slice(12, 15),
      u_color: [1, 0.8, 0.2, 1],
      u_shininess: 50,
      u_texture: texInfo.texture,
    });

    }else{
      twgl.setUniforms(programInfo, {
      u_matrix: worldViewProjection,
      u_worldViewProjection: worldViewProjection,
      u_world: worldMatrix,
      u_worldInverseTranspose: m4.transpose(m4.inverse(worldMatrix)),
      u_lightWorldPosition: lightPos,
      u_viewWorldPosition: cameraMatrix.slice(12, 15),
      u_color: [1, 0.8, 0.2, 1],
      u_shininess: 50,
      u_texture: texInfo.texture,
    });

    }

    // calls gl.drawArrays or gl.drawElements.
    twgl.drawBufferInfo(gl, bufferInfo, data.triangles ? gl.TRIANGLE : gl.LINES);
  }

  function getExtents(positions) {
    const min = positions.slice(0, 3);
    const max = positions.slice(0, 3);
    for (let i = 3; i < positions.length; i += 3) {
      min[0] = Math.min(positions[i + 0], min[0]);
      min[1] = Math.min(positions[i + 1], min[1]);
      min[2] = Math.min(positions[i + 2], min[2]);
      max[0] = Math.max(positions[i + 0], max[0]);
      max[1] = Math.max(positions[i + 1], max[1]);
      max[2] = Math.max(positions[i + 2], max[2]);
    }
    return {
      min: min,
      max: max,
    };
  }


  
 
  // rotates around Y axis.
  function lathePoints(points,
    startAngle, // angle to start at (ie 0)
    endAngle, // angle to end at (ie Math.PI * 2)
    numDivisions, // how many quads to make around
    capStart, // true to cap the top
    capEnd) { // true to cap the bottom
    const positions = [];
    const texcoords = [];
    const indices = [];

    const vOffset = capStart ? 1 : 0;
    const pointsPerColumn = points.length + vOffset + (capEnd ? 1 : 0);
    const quadsDown = pointsPerColumn - 1;

    // generate v coordniates
    let vcoords = [];

    // first compute the length of the points
    let length = 0;
    for (let i = 0; i < points.length - 1; ++i) {
      vcoords.push(length);
      length += v2.distance(points[i], points[i + 1]);
    }
    vcoords.push(length); // the last point

    // now divide each by the total length;
    vcoords = vcoords.map(v => v / length);

    // generate points
    for (let division = 0; division <= numDivisions; ++division) {
      const u = division / numDivisions;
      const angle = lerp(startAngle, endAngle, u) % (Math.PI * 2);
      const mat = m4.yRotation(angle);
      if (capStart) {
        // add point on Y access at start
        positions.push(0, points[0][1], 0);
        texcoords.push(u, 0);
      }
  
      points.forEach((p, ndx) => {
        const tp = m4.transformPoint(mat, [...p, 0]);
        positions.push(tp[0], tp[1], tp[2]);
        texcoords.push(u, vcoords[ndx]);
      });
      if (capEnd) {
        // add point on Y access at end
        positions.push(0, points[points.length - 1][1], 0);
        texcoords.push(u, 1);
      }
    }

    // generate indices
    for (let division = 0; division < numDivisions; ++division) {
      const column1Offset = division * pointsPerColumn;
      const column2Offset = column1Offset + pointsPerColumn;
      for (let quad = 0; quad < quadsDown; ++quad) {
        indices.push(column1Offset + quad, column1Offset + quad + 1, column2Offset + quad);
        indices.push(column1Offset + quad + 1, column2Offset + quad + 1, column2Offset + quad);
      }
    }

    return {
      position: positions,
      texcoord: texcoords,
      indices: indices,
    };
  }

  function makeIndexedIndicesFn(arrays) {
    const indices = arrays.indices;
    let ndx = 0;
    const fn = function() {
      return indices[ndx++];
    };
    fn.reset = function() {
      ndx = 0;
    };
    fn.numElements = indices.length;
    return fn;
  }

  function makeUnindexedIndicesFn(arrays) {
    let ndx = 0;
    const fn = function() {
      return ndx++;
    };
    fn.reset = function() {
      ndx = 0;
    };
    fn.numElements = arrays.positions.length / 3;
    return fn;
  }

  function makeIndiceIterator(arrays) {
    return arrays.indices ?
      makeIndexedIndicesFn(arrays) :
      makeUnindexedIndicesFn(arrays);
  }

  function generateNormals(arrays, maxAngle) {
    const positions = arrays.position;
    const texcoords = arrays.texcoord;

    // first compute the normal of each face
    let getNextIndex = makeIndiceIterator(arrays);
    const numFaceVerts = getNextIndex.numElements;
    const numVerts = arrays.position.length;
    const numFaces = numFaceVerts / 3;
    const faceNormals = [];

    // Compute the normal for every face.
    // While doing that, create a new vertex for every face vertex
    for (let i = 0; i < numFaces; ++i) {
      const n1 = getNextIndex() * 3;
      const n2 = getNextIndex() * 3;
      const n3 = getNextIndex() * 3;

      const v1 = positions.slice(n1, n1 + 3);
      const v2 = positions.slice(n2, n2 + 3);
      const v3 = positions.slice(n3, n3 + 3);

      faceNormals.push(m4.normalize(m4.cross(m4.subtractVectors(v1, v2), m4.subtractVectors(v3, v2))));
    }

    let tempVerts = {};
    let tempVertNdx = 0;

    // this assumes vertex positions are an exact match

    function getVertIndex(x, y, z) {

      const vertId = x + "," + y + "," + z;
      const ndx = tempVerts[vertId];
      if (ndx !== undefined) {
        return ndx;
      }
      const newNdx = tempVertNdx++;
      tempVerts[vertId] = newNdx;
      return newNdx;
    }

    // We need to figure out the shared vertices.
    // It's not as simple as looking at the faces (triangles)
    // because for example if we have a standard cylinder
    //
    //
    //      3-4
    //     /   \
    //    2     5   Looking down a cylinder starting at S
    //    |     |   and going around to E, E and S are not
    //    1     6   the same vertex in the data we have
    //     \   /    as they don't share UV coords.
    //      S/E
    //
    // the vertices at the start and end do not share vertices
    // since they have different UVs but if you don't consider
    // them to share vertices they will get the wrong normals

    const vertIndices = [];
    for (let i = 0; i < numVerts; ++i) {
      const offset = i * 3;
      const vert = positions.slice(offset, offset + 3);
      vertIndices.push(getVertIndex(vert));
    }

    // go through every vertex and record which faces it's on
    const vertFaces = [];
    getNextIndex.reset();
    for (let i = 0; i < numFaces; ++i) {
      for (let j = 0; j < 3; ++j) {
        const ndx = getNextIndex();
        const sharedNdx = vertIndices[ndx];
        let faces = vertFaces[sharedNdx];
        if (!faces) {
          faces = [];
          vertFaces[sharedNdx] = faces;
        }
        faces.push(i);
      }
    }

    // now go through every face and compute the normals for each
    // vertex of the face. Only include faces that aren't more than
    // maxAngle different. Add the result to arrays of newPositions,
    // newTexcoords and newNormals, discarding any vertices that
    // are the same.
    tempVerts = {};
    tempVertNdx = 0;
    const newPositions = [];
    const newTexcoords = [];
    const newNormals = [];

    function getNewVertIndex(x, y, z, nx, ny, nz, u, v) {
      const vertId =
        x + "," + y + "," + z + "," +
        nx + "," + ny + "," + nz + "," +
        u + "," + v;

      const ndx = tempVerts[vertId];
      if (ndx !== undefined) {
        return ndx;
      }
      const newNdx = tempVertNdx++;
      tempVerts[vertId] = newNdx;
      newPositions.push(x, y, z);
      newNormals.push(nx, ny, nz);
      newTexcoords.push(u, v);
      return newNdx;
    }

    const newVertIndices = [];
    getNextIndex.reset();
    const maxAngleCos = Math.cos(maxAngle);
    // for each face
    for (let i = 0; i < numFaces; ++i) {
      // get the normal for this face
      const thisFaceNormal = faceNormals[i];
      // for each vertex on the face
      for (let j = 0; j < 3; ++j) {
        const ndx = getNextIndex();
        const sharedNdx = vertIndices[ndx];
        const faces = vertFaces[sharedNdx];
        const norm = [0, 0, 0];
        faces.forEach(faceNdx => {
          // is this face facing the same way
          const otherFaceNormal = faceNormals[faceNdx];
          const dot = m4.dot(thisFaceNormal, otherFaceNormal);
          if (dot > maxAngleCos) {
            m4.addVectors(norm, otherFaceNormal, norm);
          }
        });
        m4.normalize(norm, norm);
        const poffset = ndx * 3;
        const toffset = ndx * 2;
        newVertIndices.push(getNewVertIndex(
          positions[poffset + 0], positions[poffset + 1], positions[poffset + 2],
          norm[0], norm[1], norm[2],
          texcoords[toffset + 0], texcoords[toffset + 1]));
      }
    }

    return {
      position: newPositions,
      texcoord: newTexcoords,
      normal: newNormals,
      indices: newVertIndices,
    };

  }

  webglLessonsUI.setupUI(document.querySelector("#ui"), data, [{
      type: "slider",
      key: "distance",
      change: update,
      min: 0.001,
      max: 5,
      precision: 3,
      step: 0.001,
    },
    {
      type: "slider",
      key: "divisions",
      change: update,
      min: 1,
      max: 60,
    },
    {
      type: "slider",
      key: "startAngle",
      change: update,
      min: 0,
      max: Math.PI * 2,
      precision: 3,
      step: 0.001,
      uiMult: 180 / Math.PI,
      uiPrecision: 0
    },
    {
      type: "slider",
      key: "endAngle",
      change: update,
      min: 0,
      max: Math.PI * 2,
      precision: 3,
      step: 0.001,
      uiMult: 180 / Math.PI,
      uiPrecision: 0
    },
    {
      type: "slider",
      key: "maxAngle",
      change: update,
      min: 0.001,
      max: Math.PI,
      precision: 3,
      step: 0.001,
      uiMult: 180 / Math.PI,
      uiPrecision: 1
    },
    {
      type: "checkbox",
      key: "capStart",
      change: update
    },
    {
      type: "checkbox",
      key: "capEnd",
      change: update
    },
    {
    type: "checkbox",
    key: "orthographic",
    change: update,
    },
    {
      type: "checkbox",
      key: "triangles",
      change: render
    },
    {
      type: "option",
      key: "mode",
      change: render,
      options: ["normals", "lit", "texcoords"],
    },
    {
      type: "option",
      key: "shape",
      change: update,
      options: ["shape1", "shape2", "shape3"],
    },
    { 
      type: "option", 
      key: "light", 
      change: update, 
      options: ["default","Point1","Point2","Point3","spotlight"]
    }
  ]);

  gl.canvas.addEventListener('mousedown', (e) => {
    e.preventDefault();
    startRotateCamera(e);
  });
  window.addEventListener('mouseup', stopRotateCamera);
  window.addEventListener('mousemove', rotateCamera);
  gl.canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    startRotateCamera(e.touches[0]);
  });
  window.addEventListener('touchend', (e) => {
    stopRotateCamera(e.touches[0]);
  });
  window.addEventListener('touchmove', (e) => {
    rotateCamera(e.touches[0]);
  });

  let lastPos;
  let moving;

  function startRotateCamera(e) {
    lastPos = getRelativeMousePosition(gl.canvas, e);
    moving = true;
  }

  function rotateCamera(e) {
    if (moving) {
      const pos = getRelativeMousePosition(gl.canvas, e);
      const size = [4 / gl.canvas.width, 4 / gl.canvas.height];
      const delta = v2.mult(v2.sub(lastPos, pos), size);

      // this is bad but it works for a basic case so phffttt
      worldMatrix = m4.multiply(m4.xRotation(delta[1] * 5), worldMatrix);
      worldMatrix = m4.multiply(m4.yRotation(delta[0] * 5), worldMatrix);

      lastPos = pos;

      render();
    }
  }

  function stopRotateCamera() {
    moving = false;
  }

  function degToRad(deg) {
    return deg * Math.PI / 180;
  }

  function lerp(a, b, t) {
    return a + (b - a) * t;
  }

  function getRelativeMousePosition(canvas, e) {
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / (rect.right - rect.left) * canvas.width;
    const y = (e.clientY - rect.top) / (rect.bottom - rect.top) * canvas.height;
    return [
      (x - canvas.width / 2) / window.devicePixelRatio,
      (y - canvas.height / 2) / window.devicePixelRatio,
    ];
  }

  // creates a texture info { width: w, height: h, texture: tex }
  // The texture will start with 1x1 pixels and be updated
  // when the image has loaded
  function loadImageAndCreateTextureInfo(url, callback) {
    var tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    // Fill the texture with a 1x1 blue pixel.
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
      new Uint8Array([0, 0, 255, 255]));

    var textureInfo = {
      width: 1, // we don't know the size until it loads
      height: 1,
      texture: tex,
    };
    var img = new Image();
    img.addEventListener('load', function() {
      textureInfo.width = img.width;
      textureInfo.height = img.height;

      gl.bindTexture(gl.TEXTURE_2D, textureInfo.texture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
      gl.generateMipmap(gl.TEXTURE_2D);

      if (callback) {
        callback();
      }
    });
    requestCORSIfNotSameOrigin(img, url)
    img.src = url;

    return textureInfo;
  }
}

const v2 = (function() {
  // adds 1 or more v2s
  function add(a, ...args) {
    const n = a.slice();
    [...args].forEach(p => {
      n[0] += p[0];
      n[1] += p[1];
    });
    return n;
  }

  function sub(a, ...args) {
    const n = a.slice();
    [...args].forEach(p => {
      n[0] -= p[0];
      n[1] -= p[1];
    });
    return n;
  }

  function mult(a, s) {
    if (Array.isArray(s)) {
      let t = s;
      s = a;
      a = t;
    }
    if (Array.isArray(s)) {
      return [
        a[0] * s[0],
        a[1] * s[1],
      ];
    } else {
      return [a[0] * s, a[1] * s];
    }
  }

  function lerp(a, b, t) {
    return [
      a[0] + (b[0] - a[0]) * t,
      a[1] + (b[1] - a[1]) * t,
    ];
  }

  function min(a, b) {
    return [
      Math.min(a[0], b[0]),
      Math.min(a[1], b[1]),
    ];
  }

  function max(a, b) {
    return [
      Math.max(a[0], b[0]),
      Math.max(a[1], b[1]),
    ];
  }

  // compute the distance squared between a and b
  function distanceSq(a, b) {
    const dx = a[0] - b[0];
    const dy = a[1] - b[1];
    return dx * dx + dy * dy;
  }

  // compute the distance between a and b
  function distance(a, b) {
    return Math.sqrt(distanceSq(a, b));
  }

  // compute the distance squared from p to the line segment
  // formed by v and w
  function distanceToSegmentSq(p, v, w) {
    const l2 = distanceSq(v, w);
    if (l2 === 0) {
      return distanceSq(p, v);
    }
    let t = ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2;
    t = Math.max(0, Math.min(1, t));
    return distanceSq(p, lerp(v, w, t));
  }

  // compute the distance from p to the line segment
  // formed by v and w
  function distanceToSegment(p, v, w) {
    return Math.sqrt(distanceToSegmentSq(p, v, w));
  }

  return {
    add: add,
    sub: sub,
    max: max,
    min: min,
    mult: mult,
    lerp: lerp,
    distance: distance,
    distanceSq: distanceSq,
    distanceToSegment: distanceToSegment,
    distanceToSegmentSq: distanceToSegmentSq,
  };
}());

main();


// This is needed if the images are not on the same domain
// NOTE: The server providing the images must give CORS permissions
// in order to be able to use the image with WebGL. Most sites
// do NOT give permission.
// See: http://webgl2fundamentals.org/webgl/lessons/webgl-cors-permission.html
function requestCORSIfNotSameOrigin(img, url) {
  if ((new URL(url, window.location.href)).origin !== window.location.origin) {
    img.crossOrigin = "";
  }
}
