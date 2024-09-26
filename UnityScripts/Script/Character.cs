using System.Collections;
using System.Collections.Generic;
using UnityEngine;
//create a character class
public class Character : MonoBehaviour
{
    public float speed = 10.0f;
    public float jumpSpeed = 10.0f;
    public float gravity = 20.0f;
    public float rotateSpeed = 10.0f;
    public enum CharacterState
    {
        idle,
        walk,
        run,
        jump,
        attack,
        gethit,
        die
    }
    public enum MoveDirection
    {
        forward,
        backward,
        left,
        right,
        jump
    }
    public CharacterState currentState;
    private void Start()
    {
        //set the character to the start position
        transform.position = new Vector3(0, 0, 0);
    }
    public void Move(MoveDirection direction, float amount)
    {
        //move the character
        switch (direction)
        {
            case MoveDirection.forward:
                transform.Translate(Vector3.forward * amount * Time.deltaTime);
                break;
            case MoveDirection.backward:
                transform.Translate(Vector3.back * amount * Time.deltaTime);
                break;
            case MoveDirection.left:
                transform.Translate(Vector3.left * amount * Time.deltaTime);
                break;
            case MoveDirection.right:
                transform.Translate(Vector3.right * amount * Time.deltaTime);
                break;
            case MoveDirection.jump:
                transform.Translate(Vector3.up * amount * Time.deltaTime);
                break;
            default:
                //change character state to run if ammount is greater than 0
                if (amount > 0)
                {
                    currentState = CharacterState.run;
                }
                //change character state to walk if ammount is less than 0
                else if (amount < 0)
                {
                    currentState = CharacterState.walk;
                }
                break;
        }
    }
}