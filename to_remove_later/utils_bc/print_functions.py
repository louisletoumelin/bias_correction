def print_intro():

    intro = """
                                            ''' '
                                          '   ' '
                                     ''' ''' '''
                              + hs    ' '''''  '.' '   
                            'shh  ho            '   '   
                           .yhhh  hh+           ' ''  
                          /hhhs    +hhh/         
                          hhhh'     hhhh         '''
                         ohhho      +hhh:       '.  '.' 
                       'yhhh:        ohhh: ''''' ''' .  
               .+.    -hhhy.          ohhh:  '  ''''' ''
              -hhho' /hhhs'            ohhh:  ''''''''' 
             :hhhhhhyhhh+               ohhh/      .' ''
            /hhho+hhhhh:                 +hhh+    '. '.'
           +hhh+  '+hy                    /hhho     ''  
          ohhh/     '                       :hhhs'       
        'shhh:                               :yhhy-      
       gyhhhg                 Bias            'shhh/     
      hyhhyf                                   +hhhs'   
     :hhhs'             Correction              -hhhh:  
    +hhho                                       'ohhhsg
    hhh/                                          :yhhh
    hy-                                              '+hh
    o'              by Louis Le Toumelin               .s

                     CEN - Meteo-France
    """
    print(intro)


def print_headline(name, value):
    print(f"\n\n_______________________", flush=True)
    print(f"_____{name}:{value}_____", flush=True)
    print(f"_______________________\n\n", flush=True)