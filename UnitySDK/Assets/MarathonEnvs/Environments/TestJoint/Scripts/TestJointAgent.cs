using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using MLAgents;

public class TestJointAgent : Agent {

    public float FixedDeltaTime = 0.005f;
    public List<TestJointMuscle> Muscles;
	List<float> _previousActions;
	List<float> _actions;
	float? _penalty;
	public List<float> Rewards;
	JointTrainerAgent _jointTrainerAgent;
	float _posDistance; // distance above
    float _negDistance; // distance below
	bool _isStill;
	bool _isOnTarget;
	public bool ShowMonitor;
	bool _hasValidModel;
	Dictionary<GameObject, Vector3> transformsPosition;
	Dictionary<GameObject, Quaternion> transformsRotation;
	TestJointTrainerDecision _testJointTrainerDecision;

    
    public override void AgentReset()
    {
		if (_testJointTrainerDecision != null)
			_testJointTrainerDecision.SetReward(this.GetCumulativeReward());
		_testJointTrainerDecision = FindObjectOfType<TestJointTrainerDecision>();
		ResetModel();
		if (ShowMonitor) 
			Monitor.SetActive(true);
		Time.fixedDeltaTime = FixedDeltaTime;
		_negDistance = 0f;
		_posDistance = 0f;
		_isStill = false;
		_isOnTarget = false;
		_jointTrainerAgent = GetComponent<JointTrainerAgent>();
		_previousActions = null;
		_actions = null;
		_penalty = null;
		Muscles = new List<TestJointMuscle> ();
		var muscles = GetComponentsInChildren<ConfigurableJoint>();
		ConfigurableJoint rootConfigurableJoint = null;
		var ragDoll = GetComponent<TestJointHelper>();
		Rewards = new List<float>();
		foreach (var m in muscles)
		{
			var maximumForce = new Vector3(ragDoll.MusclePowers.First(x=>x.Muscle == m.name).Power,0,0);
			// maximumForce *= 2f;
			var muscle = new TestJointMuscle{
				// Rigidbody = m.GetComponent<Rigidbody>(),
				Transform = m.GetComponent<Transform>(),
				ConfigurableJoint = m,
				Name = m.name,
				Group = TestJointBodyHelper.GetMuscleGroup(m.name),
				MaximumForce = maximumForce
			};
			if (muscle.Group == TestJointBodyHelper.MuscleGroup.Hips)
				rootConfigurableJoint = muscle.ConfigurableJoint;
            if (rootConfigurableJoint == null)
                rootConfigurableJoint = muscle.ConfigurableJoint; //HACK
			// muscle.RootConfigurableJoint = rootConfigurableJoint;
			muscle.Init();

			Muscles.Add(muscle);			
			Rewards.Add(0);		
		}
        // _assaultCourse004TerrainAgent.Terminate(GetCumulativeReward());
    }
	void ResetModel()
	{
		Transform[] allChildren = GetComponentsInChildren<Transform>();
		if (_hasValidModel)
		{
			// restore
			foreach (Transform child in allChildren)
			{
				child.position = transformsPosition[child.gameObject];
				child.rotation = transformsRotation[child.gameObject];
				var childRb = child.GetComponent<Rigidbody>();
				if (childRb != null)
				{
					childRb.angularVelocity = Vector3.zero;
					childRb.velocity = Vector3.zero;
				}
			}
			return;
		}
		allChildren = GetComponentsInChildren<Transform>();
		transformsPosition = new Dictionary<GameObject, Vector3>();
		transformsRotation = new Dictionary<GameObject, Quaternion>();
		foreach (Transform child in allChildren)
		{
			transformsPosition[child.gameObject] = child.position;
			transformsRotation[child.gameObject] = child.rotation;
		}
		_hasValidModel = true;
	}	
	public override void AgentAction(float[] vectorAction, string textAction)
	{
		int i = 0;
		foreach (var m in Muscles)
		{
            m.TargetAngularVelocityX = vectorAction[i++];
            m.UpdateMotor();
        }
		_actions = vectorAction.ToList();
		ProcessReward();
    }
	bool IsSameAction(float[] vectorAction)
	{
		if (_previousActions == null)
			return false;
		bool isSame = true;
		for (int i = 0; i < vectorAction.Length; i++)
			isSame &= (vectorAction[i] == _previousActions[i]);
		return isSame;
	}

    public override void CollectObservations()
    {
		foreach (var m in Muscles)
		{
            m.UpdateObservations();
            var diff = m.ObsNormalizedAngleX - m.TargetNormalizedAngleX;
            _posDistance = Mathf.Clamp(diff,0f,1f);
            _negDistance = Mathf.Clamp(-diff,0f,1f);
			_isStill = m.ObsRotationVelocity > -1e-2 && m.ObsRotationVelocity < 1e-2;
			_isOnTarget = diff > -1e-2 && diff < 1e-2;
            // AddVectorObs(_posDistance);
            // AddVectorObs(_negDistance);
			AddVectorObs(diff);
            AddVectorObs(m.TargetAngularVelocityX); // last action
            AddVectorObs(m.ObsNormalizedAngleX);
            AddVectorObs(m.TargetNormalizedAngleX); // target
            // AddVectorObs(m.ObsRotationVelocity);
            // AddVectorObs(_isStill);
            // AddVectorObs(_isOnTarget);
			AddVectorObs(Rewards); // Last Reward

			if (ShowMonitor) 
			{
				Monitor.Log("diff", diff);
				Monitor.Log("m.TargetAngularVelocityX", m.TargetAngularVelocityX);
				Monitor.Log("m.ObsNormalizedAngleX", m.ObsNormalizedAngleX);
				Monitor.Log("m.TargetNormalizedAngleX", m.TargetNormalizedAngleX);
				Monitor.Log("m.ObsRotationVelocity", m.ObsRotationVelocity);
			}
		}
		_penalty = null;
	}

	void ProcessReward()
	{
		bool recalcPenality = _penalty == null;
		if (recalcPenality)
			_penalty = 0f;		
		int i = 0;
		if (_previousActions == null)
			_previousActions = _actions?.ToList();
		while (recalcPenality)
		{
			var difference = Mathf.Abs(_previousActions[i]-_actions[i]);
			_penalty += difference;
			i++;
			if (i>=_actions.Count)
				break;
		}
		_previousActions = _actions?.ToList();
		Rewards = new List<float>();
		var hist = new List<float> ();
		foreach (var m in Muscles)
		{
			m.UpdateObservations();
			var distReward = m.GetReward();
			var noPenalityReward = Mathf.Pow(1f-_penalty.Value,4);
	        distReward = Mathf.Pow(distReward,4);
			distReward *= noPenalityReward;

			float reward = 
				(distReward * .100f) + 
				(noPenalityReward * .0f);			
			Rewards.Add(reward);
			hist.Add(distReward);
			hist.Add(noPenalityReward);
			hist.Add(reward);
		}

		// var noPenalityReward = 1f-(Mathf.Pow(_penalty.Value, 2));
		var aveReward = Rewards.Average();
		AddReward(aveReward);
		if (ShowMonitor) 
			Monitor.Log("rewardHist", hist.ToArray());
		_jointTrainerAgent.ChildStep(aveReward);			

		// // reward = Mathf.Pow(reward, 2);
		// // hist.Add(reward);
		// // cost *= .5f;
		// hist.Add(-_penalty.Value);
		// hist.Add(_isStill ? 1f : 0f);
		// hist.Add(_isOnTarget ? 1f : 0f);

		// distReward -= _penalty.Value;
		// if (_isStill && _isOnTarget)
		// 	distReward += .5f;
		// if (_isOnTarget)
		// 	distReward += 1f- Mathf.Clamp(Mathf.Abs(Muscles[0].ObsRotationVelocity), 0f, 1f);
		// hist.Add(distReward);
		// AddReward(distReward);
		// if (ShowMonitor) 
		// 	Monitor.Log("rewardHist", hist.ToArray());
		// _jointTrainerAgent.ChildStep(distReward);			
    }
	public static bool NearlyEqual(float a, float b, float epsilon)
{
    float absA = Mathf.Abs(a);
    float absB = Mathf.Abs(b);
    float diff = Mathf.Abs(a - b);

    if (a == b)
    { // shortcut, handles infinities
        return true;
    } 
    else if (a == 0 || b == 0 || diff < float.Epsilon) 
    {
        // a or b is zero or both are extremely close to it
        // relative error is less meaningful here
        return diff < epsilon;
    }
    else
    { // use relative error
        return diff / (absA + absB) < epsilon;
    }
}

    void ObservationsDefault()
    {
    }

    float StepRewardTestBed()
    {
        return 0f;
    }
}
