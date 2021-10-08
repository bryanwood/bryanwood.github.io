const canvas = document.createElement("canvas");

canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

canvas.style.position = "fixed";
canvas.style.zIndex = -10;

document.body.appendChild(canvas);

const gl = canvas.getContext("webgl2");

const identityMatrix = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

const vertexShader = `
#version 300 es

in vec4 aVertexPosition;
in vec2 aTextureCoord;

uniform mat4 uProjectionMatrix;
uniform float uTime;

out vec2 uv;

void main(){
  gl_Position = uProjectionMatrix * aVertexPosition;
  uv = aTextureCoord;
}
`.trim();

const fragmentShader = `
#version 300 es

precision highp float;

uniform vec4 uColor;
uniform float uTime;

in vec2 uv;

out vec4 outColor;

//
// Description : Array and textureless GLSL 2D/3D/4D simplex 
//               noise functions.
//      Author : Ian McEwan, Ashima Arts.
//  Maintainer : stegu
//     Lastmod : 20201014 (stegu)
//     License : Copyright (C) 2011 Ashima Arts. All rights reserved.
//               Distributed under the MIT License. See LICENSE file.
//               https://github.com/ashima/webgl-noise
//               https://github.com/stegu/webgl-noise
// 

vec3 mod289(vec3 x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 mod289(vec4 x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 permute(vec4 x) {
     return mod289(((x*34.0)+1.0)*x);
}

vec4 taylorInvSqrt(vec4 r)
{
  return 1.79284291400159 - 0.85373472095314 * r;
}

float snoise(vec3 v)
  { 
  const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
  const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

// First corner
  vec3 i  = floor(v + dot(v, C.yyy) );
  vec3 x0 =   v - i + dot(i, C.xxx) ;

// Other corners
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min( g.xyz, l.zxy );
  vec3 i2 = max( g.xyz, l.zxy );

  //   x0 = x0 - 0.0 + 0.0 * C.xxx;
  //   x1 = x0 - i1  + 1.0 * C.xxx;
  //   x2 = x0 - i2  + 2.0 * C.xxx;
  //   x3 = x0 - 1.0 + 3.0 * C.xxx;
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
  vec3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y

// Permutations
  i = mod289(i); 
  vec4 p = permute( permute( permute( 
             i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0 )) 
           + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

// Gradients: 7x7 points over a square, mapped onto an octahedron.
// The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
  float n_ = 0.142857142857; // 1.0/7.0
  vec3  ns = n_ * D.wyz - D.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

  vec4 x = x_ *ns.x + ns.yyyy;
  vec4 y = y_ *ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4( x.xy, y.xy );
  vec4 b1 = vec4( x.zw, y.zw );

  //vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
  //vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
  vec4 s0 = floor(b0)*2.0 + 1.0;
  vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

  vec3 p0 = vec3(a0.xy,h.x);
  vec3 p1 = vec3(a0.zw,h.y);
  vec3 p2 = vec3(a1.xy,h.z);
  vec3 p3 = vec3(a1.zw,h.w);

//Normalise gradients
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;

// Mix final noise value
  vec4 m = max(0.5 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 105.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), 
                                dot(p2,x2), dot(p3,x3) ) );
  }

  vec2 tile(float t, float m, float s){
      float x = (1.0 - (t + 1.0)) * s;
      float y = cos(t) * m;
      return vec2(x,y);
  }

  float noise(vec2 uv, float ct)
  {
    vec2 step = vec2(1.3, 1.7);
    float n = snoise(vec3(uv.xy, ct));
    n += 0.5 * snoise(vec3(uv.xy * 2.0 - step, ct));
    n += 0.25 * snoise(vec3(uv.xy * 4.0 - 2.0 * step, ct));
    n += 0.125 * snoise(vec3(uv.xy * 8.0 - 3.0 * step, ct));
    n += 0.0625 * snoise(vec3(uv.xy * 16.0 - 4.0 * step, ct));
    n += 0.03125 * snoise(vec3(uv.xy * 32.0 - 5.0 * step, ct));
    return n;
  }

  vec3 shift_noise(vec3 value, float min, float range)
  {
    return vec3(min + range * value);
  }

  vec4 lower_noise(vec2 uv, float t, float ct, float steps, float min, float range)
  {
    vec2 uv2 = uv * vec2(0.18, 1.0) + tile(t, 0.06, 0.05);
    float n = noise(uv2, ct);
    n = floor(n * steps) / steps;
    return vec4(shift_noise(vec3(n), min, range), 1.0);
  }

void main(){
    float speed = 0.3;
    float t = uTime;
    float ct = speed * cos(uTime);
    float st = speed * sin(uTime);


    vec4 a = lower_noise(uv, t, ct, 12.0, 0.5, 0.25) * 0.25;
    vec4 b = lower_noise(uv, t * 2.0, st, 12.0, 0.75, 0.25) * 0.25;
    vec4 c = lower_noise(uv, t * 2.0, st, 12.0, 0.0, 0.25) * 0.25;
    vec4 d = lower_noise(uv, t, ct, 12.0, 0.25, 0.25) * 0.25;
     
    outColor = (a + b + c + d) * uColor;
}
`.trim();

const program = linkProgram(
  compileShader(gl.VERTEX_SHADER, vertexShader),
  compileShader(gl.FRAGMENT_SHADER, fragmentShader)
);

const indicies = new Uint16Array([0, 1, 2, 2, 3, 0]);

// prettier-ignore
const vertexes = new Float32Array([
    -1, -1, 0, 1, 0, 0,  
     1, -1, 0, 1, 1, 0, 
     1,  1, 0, 1, 1, 1, 
    -1,  1, 0, 1, 0, 1
]);

const vertexBuffer = gl.createBuffer();
const indexBuffer = gl.createBuffer();

const vertexPositionLocation = gl.getAttribLocation(program, "aVertexPosition");
const uvLocation = gl.getAttribLocation(program, "aTextureCoord");
const projectionMatrixLocation = gl.getUniformLocation(program, "uProjectionMatrix");
const timeLocation = gl.getUniformLocation(program, "uTime");
const colorLocation = gl.getUniformLocation(program, "uColor");

const vao = gl.createVertexArray();

gl.bindVertexArray(vao);
gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);

gl.enableVertexAttribArray(vertexPositionLocation);
gl.vertexAttribPointer(vertexPositionLocation, 4, gl.FLOAT, false, 6 * 4, 0 * 4);

gl.enableVertexAttribArray(uvLocation);
gl.vertexAttribPointer(uvLocation, 2, gl.FLOAT, false, 6 * 4, 4 * 4);

gl.useProgram(program);

gl.uniformMatrix4fv(projectionMatrixLocation, false, identityMatrix);

gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
gl.bufferData(gl.ARRAY_BUFFER, vertexes, gl.DYNAMIC_DRAW);
gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indicies, gl.DYNAMIC_DRAW);

function color(t) {
  return [Math.sin(t) * 0.5 + 0.5, Math.sin(t * 2) * 0.5 + 0.5, Math.sin(t / 2) * 0.5 + 0.5, 1];
}

function draw(time) {
  gl.uniform4fv(colorLocation, color(time));
  gl.uniform1f(timeLocation, time);
  gl.drawElements(gl.TRIANGLES, 3 * 2, gl.UNSIGNED_SHORT, 0);
}

function compileShader(type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error(gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

function linkProgram(vertexShader, fragmentShader) {
  const program = gl.createProgram();
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error(gl.getProgramInfoLog(program));
    gl.deleteProgram(program);
    return null;
  }
  return program;
}

function loop(time) {
  requestAnimationFrame(loop);
  draw(time / 1000);
}

requestAnimationFrame(loop);
