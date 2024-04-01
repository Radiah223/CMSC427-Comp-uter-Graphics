/** @type {HTMLCanvasElement} */
const canvas = document.querySelector('#c');
//setting canvas to the window size
canvas.width = window.innerWidth; 
canvas.height = window.innerHeight; 
//Getting access to WebGL 
/** @type {WebGLRenderingContext} */
const gl = canvas.getContext("webgl2"); 
if(!gl){ 
}
//Step three: Compile vertex and fragment shader source code in GLSL 
var vertexShaderSource =`#version 300 es

// an attribute is an input (in) to a vertex shader.
// It will receive data from a buffer
in vec2 a_position;

// Used to pass in the resolution of the canvas

//uniform: fixed value
uniform vec2 u_resolution;

// all shaders have a main function
void main() {

  // convert the position from pixels to 0.0 to 1.0
  vec2 zeroToOne = (a_position) / u_resolution;

  // convert from 0->1 to 0->2
  vec2 zeroToTwo = zeroToOne * 2.0;

  // convert from 0->2 to -1->+1 (clipspace)
  vec2 clipSpace = zeroToTwo - 1.0;

  gl_Position = vec4(clipSpace*vec2(1,-1), 0, 1);
}
`;

var fragmentShaderSource = `#version 300 es

precision highp float;

//###setting color via uniform 
uniform vec4 u_color;

// we need to declare an output for the fragment shader
out vec4 outColor;

void main() {
  outColor = u_color;
}
`;;


function createShader(gl, type, source) {
  var shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  var success = gl.getShaderParameter(shader, gl.COMPILE_STATUS);
  if (success) {
    return shader;
  }

  console.log(gl.getShaderInfoLog(shader));  
  gl.deleteShader(shader);
  return undefined;
}
var vertexShader = createShader(gl,gl.VERTEX_SHADER,vertexShaderSource); 
var fragmentShader = createShader(gl,gl.FRAGMENT_SHADER,fragmentShaderSource); 

//Step five: Link the shaders to a program 

//creating program 
function createProgram(gl, vertexShader, fragmentShader) {
  var program = gl.createProgram();
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  var success = gl.getProgramParameter(program, gl.LINK_STATUS);
  if (success) {
    return program;
  }
 
  console.log(gl.getProgramInfoLog(program));
  gl.deleteProgram(program);
}
var program = createProgram(gl,vertexShader,fragmentShader);

//converts input pixel to clipspace 
var resolutionUniformLocation = gl.getUniformLocation(program, "u_resolution");

//Step 6: Get Attribute and color location 
var positionAttributeLocation = gl.getAttribLocation(program, "a_position"); 
var colorLocation = gl.getUniformLocation(program, "u_color");



//creating a buffer in order to retrieve data 
var positionBuffer = gl.createBuffer(); 

//creating a binding point to refer to the resource 
gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer); 

//Step 7: Input Data through binding point 

// ##### function for generating the positions of an octagon based on the center point with six triangles 
function createOctagon(x_center,y_center,x_scale,y_scale){ 
  return [
    x_center,y_center,
    x_center+x_scale,y_center-y_scale, 
    x_center-x_scale,y_center-y_scale, 

    x_center+x_scale,y_center-y_scale,
    x_center,y_center, 
    x_center+2*x_scale, y_center,

    x_center-x_scale,y_center-y_scale,
    x_center,y_center, 
    x_center-2*x_scale,y_center,

    x_center,y_center,
    x_center+x_scale,y_center+y_scale, 
    x_center-x_scale,y_center+y_scale, 

    x_center,y_center,
    x_center-2*x_scale,y_center,
    x_center-x_scale,y_center+y_scale,
    

    x_center,y_center, 
    x_center+2*x_scale,y_center,
    x_center+x_scale,y_center+y_scale,
    ]; 
}
var positions = createOctagon(750,350,90,155); 


//Note: static draw means that data will not chance 
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);

//Step 8: Telling how to get data through VAO 

var vao = gl.createVertexArray(); 
gl.bindVertexArray(vao); 

//Telling webgl to get data out 
gl.enableVertexAttribArray(positionAttributeLocation);

//Telling how to pull data out 
var size = 2;          // 2 components per iteration
var type = gl.FLOAT;   // the data is 32bit floats
var normalize = false; // don't normalize the data
var stride = 0;        // 0 = move forward size * sizeof(type) each iteration to get the next position
var offset = 0;        // start at the beginning of the buffer
gl.vertexAttribPointer(positionAttributeLocation, size, type, normalize, stride, offset); 

webglUtils.resizeCanvasToDisplaySize(gl.canvas);

gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
// Clear the canvas
gl.clearColor(0, 0, 0, 0);
gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

gl.useProgram(program);
gl.uniform2f(resolutionUniformLocation, gl.canvas.width, gl.canvas.height);

//##### Setting color to dark red 
gl.uniform4f(colorLocation, 128 /255 ,5 /255 ,5 /255, 1.0);

//##### Drawing the actual octagon 
gl.bindVertexArray(vao);
var primitiveType = gl.TRIANGLES;
var offset = 0;
var count = 18;
gl.drawArrays(primitiveType, offset, count);

//##### Adding a new shape (the letter O) to the canvas 

//creating a new buffer 
var new_buffer = gl.createBuffer(); 
gl.bindBuffer(gl.ARRAY_BUFFER,new_buffer);

//function to rotate a point
function rotate(base,angle){
  var radians = (Math.PI / 180) * angle;
  var new_pos = []; 
  for (let i = 0; i< base.length; i++){ 
    if(i % 2 == 0){ 
      new_pos[i] = Math.round(base[i] * Math.cos(radians) - base[i+1] * Math.sin(radians));
    }else{ 
      new_pos[i] = Math.round(base[i-1] * Math.sin(radians) + base[i] * Math.cos(radians));
    }
  }
  console.log(new_pos);
  return new_pos; 
}

//function to create a trapezoid
function createTrapezoid(x_center,y_center,x_scale,y_scale){ 
  return [   
    x_center,y_center,
    x_center+x_scale,y_center-y_scale, 
    x_center-x_scale,y_center-y_scale, 

    x_center+x_scale,y_center-y_scale,
    x_center,y_center, 
    x_center+2*x_scale, y_center,

    x_center-x_scale,y_center-y_scale,
    x_center,y_center, 
    x_center-2*x_scale,y_center
    ]; 
}

//####creating the letter O using four trapezoids 
var base_trapezoid = createTrapezoid(350,-790,40,40);
var rotate_trap = rotate(base_trapezoid,90);
var base_trapezoid = createTrapezoid(-350,710,40,40);
var rotate_trap2 = rotate(base_trapezoid,-90); 
var rotate_trap3 = rotate(createTrapezoid(-750,-390,40,40),180);
var new_positions = createTrapezoid(750,310,40,40);
var pos = new_positions.concat(rotate_trap,rotate_trap2,rotate_trap3); 


console.log(pos.length);
gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(pos),gl.STATIC_DRAW);
var new_vao = gl.createVertexArray(); 
gl.bindVertexArray(new_vao); 
gl.enableVertexAttribArray(positionAttributeLocation); 

//pulling out the data for the second shape
var size = 2;          // 2 components per iteration
var type = gl.FLOAT;   // the data is 32bit floats
var normalize = false; // don't normalize the data
var stride = 0;        // 0 = move forward size * sizeof(type) each iteration to get the next position
var offset = 0;        // start at the beginning of the buffer
gl.vertexAttribPointer(positionAttributeLocation, size, type, normalize, stride, offset); 
gl.useProgram(program);

//##### Setting color to  black
gl.uniform4f(colorLocation, 0 ,0 ,0, 1.0);

//##### Drawing the letter
gl.bindVertexArray(new_vao);
var primitiveType = gl.TRIANGLES;
var offset = 0;
var count = 72;
gl.drawArrays(primitiveType, offset, count);




//#### Adding a cloud using triangle strips to the canvas

//creating a new buffer 
var strip_buffer = gl.createBuffer(); 
gl.bindBuffer(gl.ARRAY_BUFFER,strip_buffer); 

//generating points for the triangle strip 
var strip_position = [
  150,150,
  200,150,
  200,150,
  250,150,
  250,100,
  280,150,
  300,80,
  330,150,
  350,140,
  350,150,
  380,150,
  380,140,
  400,150
].map(function(x) { return (x-70) * 2; }); //scaling to make the cloud look larger 

gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(strip_position),gl.STATIC_DRAW);
//creating new vao object 
var strip_vao = gl.createVertexArray(); 
gl.bindVertexArray(strip_vao); 
gl.enableVertexAttribArray(positionAttributeLocation); 

var size = 2;          // 2 components per iteration
var type = gl.FLOAT;   // the data is 32bit floats
var normalize = false; // don't normalize the data
var stride = 0;        // 0 = move forward size * sizeof(type) each iteration to get the next position
var offset = 0;        // start at the beginning of the buffer
gl.vertexAttribPointer(positionAttributeLocation, size, type, normalize, stride, offset); 
gl.useProgram(program);

//setting the color as light blue 
gl.uniform4f(colorLocation, 179/255, 230/255, 245/255, 1.0);
gl.bindVertexArray(strip_vao);
var primitiveType = gl.TRIANGLE_STRIP;
var offset = 0;
var count = 13;
gl.drawArrays(primitiveType, offset, count);










