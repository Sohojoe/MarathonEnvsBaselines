using System.Collections.Generic;
using System.Linq;
using MLAgents;
using UnityEngine;

public class TestJointDecision : MonoBehaviour, Decision
{
    Brain _brain;

    public float[] Actions;
    public bool ApplyRandomActions;
    void Start()
    {
        _brain = GetComponent<Brain>();
        Actions = Enumerable.Repeat(0f, _brain.brainParameters.vectorActionSize[0]).ToArray();
    }

    public float[] Decide(
        List<float> vectorObs,
        List<Texture2D> visualObs,
        float reward,
        bool done,
        List<float> memory)
    {
        float posDistance = vectorObs[0];
        float negDistance = vectorObs[1];
        float targetAngularVelocityX = vectorObs[2];
        float obsNormalizedAngleX = vectorObs[3];
        float targetNormalizedAngleX = vectorObs[4];
        float obsRotationVelocity = vectorObs[5];
        

        if (posDistance > 0.001f) {
            if (targetAngularVelocityX < 0)
                targetAngularVelocityX = 0;
            else
                targetAngularVelocityX = 0.1f;
        }
        else if (negDistance > 0.001f){
            if (targetAngularVelocityX > 0)
                targetAngularVelocityX = 0;
            else
                targetAngularVelocityX = 0.05f;
        }
        Actions[0] = targetAngularVelocityX;

        if (ApplyRandomActions)
        {
            for (int i = 0; i < Actions.Length; i++)
                Actions[i] = Random.value * 2 - 1;
        }

        return Actions;
    }

    public List<float> MakeMemory(
        List<float> vectorObs,
        List<Texture2D> visualObs,
        float reward,
        bool done,
        List<float> memory)
    {
        return new List<float>();
    }
}