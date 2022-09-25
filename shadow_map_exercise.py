# กลุ่ม: 9
# ชื่อกลุ่ม: Group
# 6310400975 ณัฐกฤษฎิ์ พรรณจักร
# 6310406302 ธนินท์พัชร์ ศรีขันแก้ว
import sys, os
from OpenGL.GLUT import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import pandas as pd
from PIL import Image
import gl_helpers as glh

win_w, win_h = 1280, 960
t_value, wireframe, pause = 0, False, True
n_vertices, positions, colors, normals, uvs = 0, None, None, None, None
centroid, bbox, t_value = None, None, 0
mouse = [0, 0, GLUT_LEFT_BUTTON, GLUT_UP]
rotate_degree = [0, 0, 0]
shadow_map_size = 1024

def save_depth_map():
    data = glReadPixels(0, 0, shadow_map_size, shadow_map_size, GL_DEPTH_COMPONENT, GL_FLOAT)
    data3 = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.uint8)
    data3[:,:,0] = 255*data[:]
    data3[:,:,1] = 255*data[:]
    data3[:,:,2] = 255*data[:]
    image = Image.frombytes(mode="RGB", size=(shadow_map_size, shadow_map_size), data=data3)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image.save("depth_map.jpg")

def idle():
    global t_value
    t_value += 0.01
    glutPostRedisplay()

def print_shader_info_log(shader, prompt=""):
    result = glGetShaderiv(shader, GL_COMPILE_STATUS)
    if not result:
        print("%s: %s" % (prompt, glGetShaderInfoLog(shader).decode("utf-8")))
        os._exit()

def print_program_info_log(shader, prompt=""):
    result = glGetProgramiv(shader, GL_LINK_STATUS)
    if not result:
        print("%s: %s" % (prompt, glGetProgramInfoLog(shader).decode("utf-8")))
        os._exit()

def compile_program(vert_code, frag_code):
    vert_id = glCreateShader(GL_VERTEX_SHADER)
    frag_id = glCreateShader(GL_FRAGMENT_SHADER)

    glShaderSource(vert_id, vert_code)
    glShaderSource(frag_id, frag_code)

    glCompileShader(vert_id)
    glCompileShader(frag_id)
    print_shader_info_log(vert_id, "Vertex Shader")
    print_shader_info_log(frag_id, "Fragment Shader")

    prog_id = glCreateProgram()
    glAttachShader(prog_id, vert_id)
    glAttachShader(prog_id, frag_id)

    glLinkProgram(prog_id)
    print_program_info_log(prog_id, "Link Error")
    return prog_id

def motion_func(x ,y):
    dx, dy = x-mouse[0], y-mouse[1]
    button, state = mouse[2], mouse[3]
    mouse[0], mouse[1] = x, y
    if state == GLUT_DOWN:
        if button == GLUT_LEFT_BUTTON:
            if abs(dx) > abs(dy):
                rotate_degree[0] += dx
            else:
                rotate_degree[1] += dy
        elif button == GLUT_MIDDLE_BUTTON:
            if abs(dx) > abs(dy):
                rotate_degree[2] += dx
            else:
                rotate_degree[2] += dy
    glutPostRedisplay()

def mouse_func(button, state, x, y):
    mouse[0], mouse[1], mouse[2], mouse[3] = x, y, button, state
    glutPostRedisplay()

def reshape(w, h):
    global win_w, win_h, proj_mat

    win_w, win_h = w, h
    glViewport(0, 0, w, h)  
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    proj_mat = glh.Perspective(60, win_w/win_h, 0.01, 100)

def keyboard(key, x, y):
    global wireframe, pause

    key = key.decode("utf-8")
    if key == ' ':
        pause = not pause
        glutIdleFunc(None if pause else idle)
    elif key == 'w':
        wireframe = not wireframe
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE if wireframe else GL_FILL)
    elif key == 'q':
        os._exit(0)
    glutPostRedisplay()

def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    view_mat = glh.LookAt(centroid[0], centroid[1], centroid[2]+0.5*max(bbox), *centroid, 0, 1, 0)
    model_mat = glh.Rotate(rotate_degree[0], 0, 1, 0)
    model_mat = model_mat @ glh.Rotate(rotate_degree[1], 1, 0, 0)
    model_mat = model_mat @ glh.Rotate(rotate_degree[2], 0, 0, 1)

    glUseProgram(shadow_prog_id)
    glActiveTexture(GL_TEXTURE0)
    glUseProgram(shadow_prog_id)
    light_position = [2, 8, 20]
    light_at = [-3.5, 2.3, -0.2]
    light_proj_mat = glh.Perspective(60, 1, 20, 60)
    light_view_mat = glh.LookAt(*light_position, *light_at, 0, 1, 0)
    MVP = light_proj_mat @ light_view_mat @ model_mat
    glUniformMatrix4fv(glGetUniformLocation(shadow_prog_id, "MVP"), 1, True, MVP)

    glDrawBuffer(GL_NONE)
    glBindTexture(GL_TEXTURE_2D, 0)
    glViewport(0, 0, shadow_map_size, shadow_map_size)
    glBindFramebuffer(GL_FRAMEBUFFER, shadow_FBO)
    glClear(GL_DEPTH_BUFFER_BIT)
    glBindVertexArray(shadow_vao)
    glDrawArrays(GL_TRIANGLES, 0, n_vertices)
    save_depth_map()

    glUseProgram(render_prog_id)
    B = np.array(((0.5, 0, 0, 0.5), 
                  (0, 0.5, 0, 0.5),
                  (0, 0, 0.5, 0.5),
                  (0, 0, 0, 1)), dtype=np.float32)
    shadow_mat = B @ light_proj_mat @ light_view_mat @ model_mat    
    glUniformMatrix4fv(glGetUniformLocation(render_prog_id, "model_mat"), 1, True, model_mat)
    glUniformMatrix4fv(glGetUniformLocation(render_prog_id, "view_mat"), 1, True, view_mat)
    glUniformMatrix4fv(glGetUniformLocation(render_prog_id, "proj_mat"), 1, True, proj_mat)
    glUniformMatrix4fv(glGetUniformLocation(render_prog_id, "shadow_mat"), 1, True, shadow_mat)
    glUniform3fv(glGetUniformLocation(render_prog_id, "light_position"), 1, light_position)
    glUniform3fv(glGetUniformLocation(render_prog_id, "light_intensity"), 1, [1, 1, 1])
    glUniform3fv(glGetUniformLocation(render_prog_id, "Ka"), 1, [0.05, 0.05, 0.05])
    glUniform3fv(glGetUniformLocation(render_prog_id, "Ks"), 1, [1.0, 1.0, 1.0])
    glUniform1f(glGetUniformLocation(render_prog_id, "shininess"), 50.0)

    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glDrawBuffer(GL_BACK)
    glViewport(0, 0, win_w, win_h)
    glBindTexture(GL_TEXTURE_2D, shadow_tex_id)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glBindVertexArray(render_vao)
    glDrawArrays(GL_TRIANGLES, 0, n_vertices)
    glutSwapBuffers()

def init_shaders():
    global shadow_prog_id, render_prog_id
    global shadow_vao, render_vao

    shadow_vert_code = '''
#version 120
uniform mat4 MVP;
attribute vec3 position;
void main()
{
    gl_Position = MVP * vec4(position, 1);
}
'''
    shadow_frag_code = '''
#version 110
void main()
{
}
'''
    shadow_prog_id = compile_program(shadow_vert_code, shadow_frag_code)
    glUseProgram(shadow_prog_id)

    shadow_vao = glGenVertexArrays(1)
    glBindVertexArray(shadow_vao)
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, positions, GL_STATIC_DRAW)
    position_loc = glGetAttribLocation(shadow_prog_id, "position")
    if position_loc != -1:
        glVertexAttribPointer(position_loc, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
        glEnableVertexAttribArray(position_loc)

    render_vert_code = '''
#version 140
in vec3 position, color, normal;
in vec2 uv;
uniform mat4 model_mat, view_mat, proj_mat, shadow_mat;
out vec3 P, eye_position, lerped_normal, Kd;
out vec4 ss_position;   // Shadow-space (Light-space) position
void main()
{
    gl_Position = proj_mat * view_mat * model_mat * vec4(position, 1);
    P = (model_mat * vec4(position, 1)).xyz;
    eye_position = (inverse(view_mat) * vec4(0, 0, 0, 1)).xyz;
    mat4 adjunct_mat = transpose(inverse(model_mat));
    lerped_normal = (adjunct_mat * vec4(normal, 0)).xyz;
    ss_position = shadow_mat * vec4(position, 1);
    Kd = color;
}
'''
    render_frag_code = '''
#version 140
in vec3 P, eye_position, lerped_normal, Kd;
in vec4 ss_position;
uniform vec3 light_position, light_intensity, Ka, Ks;
uniform float shininess;
uniform sampler2D shadow_map;
void main()
{
    vec3 shadow_color = vec3(0.15, 0.15, 0.15);
    vec3 ambient, diffuse, specular;

    vec3 L = normalize(light_position - P);
    vec3 V = normalize(-P);
    vec3 N = normalize(lerped_normal);
    vec3 R = 2 * dot(L, N) * N - L;
    ambient = Ka * light_intensity;
    diffuse = Kd * max(dot(N, L), 0) * light_intensity;
    specular = Ks * pow(max(dot(V, R), 0), shininess) * light_intensity;
    if (dot(N, L) <= 0)
        specular = vec3(0, 0, 0); 

    vec3 color = ambient + diffuse + specular;

    float visibility = 1.0;
    float bias = max(0.05 * (1.0 - dot(N, L)), 0.005); 
    for(int i=-1; i <= 1; i++)
        for(int j=-1; j <= 1; j++)
            if (ss_position.z/ss_position.w < 1.0 && 
                textureOffset(shadow_map, ss_position.st/ss_position.w, ivec2(i, j)).r < 
                ss_position.z/ss_position.w - bias)
                visibility -= 0.1;
    gl_FragColor = vec4(color * visibility + shadow_color, 1);  
}
'''
    render_prog_id = compile_program(render_vert_code, render_frag_code)
    glUseProgram(render_prog_id)

    render_vao = glGenVertexArrays(1)
    glBindVertexArray(render_vao)
    vbo = glGenBuffers(4)
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0])
    glBufferData(GL_ARRAY_BUFFER, positions, GL_STATIC_DRAW)
    position_loc = glGetAttribLocation(render_prog_id, "position")
    if position_loc != -1:
        glVertexAttribPointer(position_loc, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
        glEnableVertexAttribArray(position_loc)

    color_loc = glGetAttribLocation(render_prog_id, "color")
    glBindBuffer(GL_ARRAY_BUFFER, vbo[1])
    glBufferData(GL_ARRAY_BUFFER, colors, GL_STATIC_DRAW)
    if color_loc != -1:
        glVertexAttribPointer(color_loc, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
        glEnableVertexAttribArray(color_loc)

    normal_loc = glGetAttribLocation(render_prog_id, "normal")
    glBindBuffer(GL_ARRAY_BUFFER, vbo[2])
    glBufferData(GL_ARRAY_BUFFER, normals, GL_STATIC_DRAW)
    if normal_loc != -1:
        glVertexAttribPointer(normal_loc, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
        glEnableVertexAttribArray(normal_loc)

    uv_loc = glGetAttribLocation(render_prog_id, "uv")
    glBindBuffer(GL_ARRAY_BUFFER, vbo[3])
    glBufferData(GL_ARRAY_BUFFER, uvs, GL_STATIC_DRAW)
    if uv_loc != -1:
        glVertexAttribPointer(uv_loc, 2, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
        glEnableVertexAttribArray(uv_loc)

    global shadow_tex_id, shadow_FBO
    glActiveTexture(GL_TEXTURE0)
    shadow_tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, shadow_tex_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, shadow_map_size, shadow_map_size, 
        0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, [1, 1, 1, 1])
    glBindTexture(GL_TEXTURE_2D, 0)
    if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
        exit(1)

    shadow_FBO = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, shadow_FBO)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
        GL_TEXTURE_2D, shadow_tex_id, 0)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    

def init_gl_and_model():
    global n_vertices, positions, colors, normals, uvs, centroid, bbox

    glClearColor(0.01, 0.01, 0.2, 0)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)

    df = pd.read_csv("../models/objects_and_walls.tri",delim_whitespace=True, 
                     comment='#', header=None, dtype=np.float32)
    centroid = df.values[:, 0:3].mean(axis=0)
    bbox = df.values[:, 0:3].max(axis=0) - df.values[:, 0:3].min(axis=0)

    positions = df.values[:, 0:3]
    colors = df.values[:, 3:6]
    normals = df.values[:, 6:9]
    uvs = df.values[:, 9:11]

    n_vertices = len(positions)
    print("no. of vertices: %d, no. of triangles: %d" % 
          (n_vertices, n_vertices//3))
    print("Centroid:", centroid)
    print("BBox:", bbox)

def main():  
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(win_w, win_h)
    glutCreateWindow("Shadow Map Exercise")
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutMouseFunc(mouse_func)
    glutPassiveMotionFunc(motion_func)
    glutMotionFunc(motion_func)
    glutIdleFunc(idle)
    init_gl_and_model()
    init_shaders()
    glutMainLoop()

if __name__ == "__main__":
    main()