def generate_cosine_fragment_shader_code(texture_count):
    return f"""
    #version 330 core
    in vec3 frag_position;
    in vec3 frag_normal;
    
    out vec4 frag_color;

    uniform sampler2DArray textures; // Array of textures
    uniform sampler2DArray depth_maps; // Depth maps for occlusion testing
    
    uniform int texture_count;
    uniform int active_textures[{texture_count}]; // Array indicating active textures
    uniform int refined_textures[{texture_count}]; // Array indicating refined textures
    layout(std140) uniform Projectors {{
        vec3 projector_directions[{texture_count}]; // Directions of the projectors
    }};

    uniform mat4 projector_view_matrices[{texture_count}]; // View matrices of the projectors
    uniform mat4 projector_proj_matrices[{texture_count}]; // Projection matrices of the projectors

    uniform float epsilon; // Epsilon for occlusion testing
    
    void main() {{
        float weight = -1.0;
        float refined_weight_factor = 1.0;
        float unrefined_weight_factor = 0.;
        float frag_depth = 0.;
        float depth_map_value = 0.;

        for (int i = 0; i < texture_count; ++i) {{
            if (active_textures[i] == 0)  continue; // Skip inactive textures
            if (refined_textures[i] == 0) continue; // Skip unrefined textures
    
            vec3 projection_dir = normalize(projector_directions[i]);
            float alignment = dot(normalize(frag_normal), projection_dir); // Calculate alignment weight

            alignment = clamp(alignment, 0.0, 1.0);

            vec4 frag_pos_clip = projector_proj_matrices[i] * projector_view_matrices[i] * vec4(frag_position, 1.0);
            vec3 frag_pos_ndc = frag_pos_clip.xyz / frag_pos_clip.w;
            vec2 frag_uv = frag_pos_clip.xy * 0.5 + 0.5;

            vec2 frag_tex_uv = frag_uv;
            frag_tex_uv.y = 1.0 - frag_tex_uv.y; // Flip the y coordinate (texture wasn't flipped when it was passed)

            frag_uv = clamp(frag_uv, 0.0, 1.0);
            frag_tex_uv = clamp(frag_tex_uv, 0.0, 1.0);

            depth_map_value = texture(depth_maps, vec3(frag_uv, i)).r;
            frag_depth = frag_pos_ndc.z;

            if ((frag_depth - depth_map_value) > epsilon) {{
                alignment = 0.;
            }}

            if (alignment < 0.3) {{
                alignment = 0.;
            }}

            vec4 tex_color = texture(textures, vec3(frag_tex_uv, i));
            if (tex_color.a < 1.) {{
                alignment *= 0.;
            }}
                
            if (alignment > weight) {{
                weight = alignment;
            }}

        }}
    
        // Map the total_weight to a [0,1] range for visualization
        // float intensity = weight * 0.5 + 0.5;

        frag_color = vec4(vec3(weight), 1.0);
    }}
    """

def generate_fragment_shader_code(texture_count):
    return f"""
    #version 330 core

    in vec3 frag_position;
    in vec3 frag_normal;

    out vec4 frag_color;

    uniform sampler2DArray textures; // Array of textures
    uniform sampler2DArray depth_maps; // Depth maps for occlusion testing

    uniform int texture_count;
    uniform int active_textures[{texture_count}]; // Array indicating active textures
    uniform int refined_textures[{texture_count}]; // Array indicating refined textures

    layout(std140) uniform Projectors {{
        vec3 projector_directions[{texture_count}]; // Directions of the projectors
    }};

    uniform mat4 projector_view_matrices[{texture_count}]; // View matrices of the projectors
    uniform mat4 projector_proj_matrices[{texture_count}]; // Projection matrices of the projectors

    uniform float epsilon; // Epsilon for occlusion testing

        void main() {{
        vec4 blended_color = vec4(0.0);
        float total_weight = 0.0;

        float refined_weight_factor = 1.0;
        float unrefined_weight_factor = 0.00000001;

        float start_rad = 0.0;
        float end_rad   = 0.0;

        for (int i = 0; i < texture_count; ++i) {{
            if (active_textures[i] == 0) continue; // Skip inactive textures

            vec3 projection_dir = normalize(projector_directions[i]);

            // Compute the dot product (cosine of the angle)
            float dotNV = dot(normalize(frag_normal), projection_dir);
        
            start_rad = 0.3;      // Start blending at 90 degrees (cos(90°) = 0)
            end_rad   = 1.0;      // End blending at 60 degrees (cos(60°) ≈ 0.5)
        
            // Clamp the dot product to [0, 1]
            dotNV = clamp(dotNV, 0.0, 1.0);
        
            // Compute the weight using smoothstep
            float weight = smoothstep(start_rad, end_rad, dotNV);

            // Adjust weight based on whether texture is refined
            if (refined_textures[i] == 1) {{
                weight *= refined_weight_factor;
            }} else {{
                weight *= unrefined_weight_factor;
            }}
        
            vec4 frag_pos_clip = projector_proj_matrices[i] * projector_view_matrices[i] * vec4(frag_position, 1.0);
            vec3 frag_pos_ndc = frag_pos_clip.xyz / frag_pos_clip.w;
            vec2 frag_uv = frag_pos_clip.xy * 0.5 + 0.5;

            vec2 frag_tex_uv = frag_uv;
            frag_tex_uv.y = 1.0 - frag_tex_uv.y; // Flip the y coordinate (texture wasn't flipped when it was passed)

            frag_uv = clamp(frag_uv, 0.0, 1.0);
            frag_tex_uv = clamp(frag_tex_uv, 0.0, 1.0);

            float depth_map_value = texture(depth_maps, vec3(frag_uv, i)).r;
            float frag_depth = frag_pos_ndc.z;

            if ((frag_depth - depth_map_value) > epsilon) {{
                weight *= 0.0;
                continue;
            }}
            
            vec4 tex_color = texture(textures, vec3(frag_tex_uv, i));
            if (tex_color.a < 1.) {{
                weight *= 0.;
            }}
            total_weight += weight;

            blended_color.rgb += tex_color.rgb * weight;

        }}

        if (total_weight > 0.0) {{
            blended_color.rgb /= total_weight;
            blended_color.a = 1.;
        }}
        frag_color = blended_color;
    }}
    """

def generate_fragment_shader_baking_code(texture_count):
    return f"""
    #version 330 core

    in vec2 frag_uv; // Ensure your vertex shader passes UV coordinates
    in vec3 frag_position;
    in vec3 frag_normal;
    out vec4 frag_color;

    uniform sampler2DArray textures; // Array of textures
    uniform sampler2DArray depth_maps; // Depth maps for occlusion testing

    uniform int texture_count;
    uniform int active_textures[{texture_count}]; // Array indicating active textures
    uniform int refined_textures[{texture_count}]; // Array indicating refined textures

    layout(std140) uniform Projectors {{
        vec3 projector_directions[{texture_count}]; // Directions of the projectors
    }};

    uniform mat4 projector_view_matrices[{texture_count}]; // View matrices of the projectors
    uniform mat4 projector_proj_matrices[{texture_count}]; // Projection matrices of the projectors

    uniform float epsilon; // Epsilon for occlusion testing

    void main() {{
        vec4 blended_color = vec4(0.0);
        float total_weight = 0.0;

        float refined_weight_factor = 1.0;
        float unrefined_weight_factor = 0.00000001;

        float start_rad = 0.0;
        float end_rad   = 0.0;

        for (int i = 0; i < texture_count; ++i) {{
            if (active_textures[i] == 0) continue; // Skip inactive textures

            vec3 projection_dir = normalize(projector_directions[i]);

            // Compute the dot product (cosine of the angle)
            float dotNV = dot(normalize(frag_normal), projection_dir);
        
            // Define the blending range (adjust 'a' and 'b' as needed)
            start_rad = 0.3;      // Start blending at 90 degrees (cos(90°) = 0)
            end_rad   = 1.0;      // End blending at 60 degrees (cos(60°) ≈ 0.5)
        
            // Clamp the dot product to [0, 1]
            dotNV = clamp(dotNV, 0.0, 1.0);
        
            // Compute the weight using smoothstep
            float weight = smoothstep(start_rad, end_rad, dotNV);

            // Adjust weight based on whether texture is refined
            if (refined_textures[i] == 1) {{
                weight *= refined_weight_factor;
            }} else {{
                weight *= unrefined_weight_factor;
            }}
        
            vec4 frag_pos_clip = projector_proj_matrices[i] * projector_view_matrices[i] * vec4(frag_position, 1.0);
            vec3 frag_pos_ndc = frag_pos_clip.xyz / frag_pos_clip.w;
            vec2 frag_uv_proj = frag_pos_ndc.xy * 0.5 + 0.5;

            vec2 frag_tex_uv = frag_uv_proj;
            frag_tex_uv.y = 1.0 - frag_tex_uv.y; // Flip the y coordinate

            frag_uv_proj = clamp(frag_uv_proj, 0.0, 1.0);
            frag_tex_uv = clamp(frag_tex_uv, 0.0, 1.0);

            float depth_map_value = texture(depth_maps, vec3(frag_uv_proj, i)).r;
            float frag_depth = frag_pos_ndc.z;

            if ((frag_depth - depth_map_value) > epsilon) {{
                weight *= 0.0;
                continue;
            }}
            
            vec4 tex_color = texture(textures, vec3(frag_tex_uv, i));
            if (tex_color.a < 1.) {{
                // if (length(tex_color.rgb) > 0.0){{
                //     weight = 0.00001;
                // }}
                // else {{
                //     weight *= 0.;
                // }}

                weight *= 0.;
            }}
            total_weight += weight;


            blended_color.rgb += tex_color.rgb * weight;

        }}

        if (total_weight > 0.0) {{
            blended_color.rgb /= total_weight;
            blended_color.a = 1.;
        }}
        frag_color = blended_color;
    }}
    """

