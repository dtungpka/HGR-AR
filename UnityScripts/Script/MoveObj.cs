using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MoveObj : MonoBehaviour
{
    // Start is called before the first frame update
    void Awake()
    {
        //transform.position = new Vector3(0, 0, 0);
    }
    void Start()
    {
        //get parent object position
        Vector3 pos = transform.parent.position;
        //check if player press A
        if (Input.GetKey(KeyCode.A))
        {
            //move object to the left
            transform.position = new Vector3(pos.x - 0.1f, pos.y, pos.z);
        }
        
    }

    // Update is called once per frame
    void Update()
    {
        //randomly move object forward
        transform.Translate(Vector3.forward * Time.deltaTime);
        
        
    }
}
