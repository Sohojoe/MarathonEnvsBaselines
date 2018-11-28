using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using MLAgents;

public class DiscreteTestJointAgent : Agent {

    public float FixedDeltaTime = 0.005f;
    public List<TestJointMuscle> Muscles;
    List<float> _previousActions;
    List<float> _previousMuscleContiniousActions;
	public List<float> MuscleContiniousActions;
	float? _penalty;
	public List<float> Rewards;
	JointTrainerAgent _jointTrainerAgent;
	float _posDistance; // distance above
    float _negDistance; // distance below
	bool _isStill;
	bool _isOnTarget;

	public float kAccelerateMod1 = 0.0001f;
	public float kAccelerateMod2 = 0.001f;
	public float kAccelerateMod3 = 0.01f;
	public bool ShowMonitor;
	bool _hasValidModel;
	Dictionary<GameObject, Vector3> transformsPosition;
	Dictionary<GameObject, Quaternion> transformsRotation;

	public List<int> RewardCount = null;

    public override void AgentReset()
    {
		ResetModel();
		if (ShowMonitor) 
			Monitor.SetActive(true);
		Time.fixedDeltaTime = FixedDeltaTime;
		_negDistance = 0f;
		_posDistance = 0f;
		_isStill = false;
		_isOnTarget = false;
		_jointTrainerAgent = GetComponent<JointTrainerAgent>();
		_previousMuscleContiniousActions = null;
		_previousActions = null;
		MuscleContiniousActions = null;
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
		MuscleContiniousActions = Enumerable.Range(0, Muscles.Count).Select(x => 0f).ToList();
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

		bool printAction = false;

		if (_previousActions == null)
			_previousActions = vectorAction.ToList();

		int i = 0;
		float cost = 0f;
		var jointAction = (int)vectorAction[i];
		var muscle = Muscles[i];
		switch (jointAction)
		{
			case 0:
				if (printAction) print("keep current speed");
				if (ShowMonitor) Monitor.Log($"action[{i}", $"action[{i}: {jointAction} - keep current speed");
				break;
			// case 1:
			// 	if (printAction) print("freeWheel");
			// 	if (ShowMonitor) Monitor.Log($"action[{i}", $"action[{i}: {jointAction} - freeWheel");
			// 	cost = MuscleContiniousActions[i] *1.4f * 3f;
			// 	MuscleContiniousActions[i] = 0f;
			// 	break;
			case 1:
				if (printAction) print("increaseSpeed by 1");
				if (ShowMonitor) Monitor.Log($"action[{i}", $"action[{i}: {jointAction} - increaseSpeed by 1");
				cost = kAccelerateMod1 * 1.1f * 3f;
				cost += _previousActions[i] >= 4 ? 1f : 0f;
				MuscleContiniousActions[i] += kAccelerateMod1;
				break;
			case 2:
				if (printAction) print("increaseSpeed by 2");
				if (ShowMonitor) Monitor.Log($"action[{i}", $"action[{i}: {jointAction} - increaseSpeed by 2");
				cost = kAccelerateMod2 * 1.2f * 3f;
				cost += _previousActions[i] >= 4 ? 1f : 0f;
				MuscleContiniousActions[i] += kAccelerateMod2;
				break;
			case 3:
				if (printAction) print("increaseSpeed by 3");
				if (ShowMonitor) Monitor.Log($"action[{i}", $"action[{i}: {jointAction} - increaseSpeed by 3");
				cost = kAccelerateMod3 * 1.3f * 3f;
				cost += _previousActions[i] >= 4 ? 1f : 0f;
				MuscleContiniousActions[i] += kAccelerateMod3;
				break;
			case 4:
				if (printAction) print("decreaseSpeed by 1");
				if (ShowMonitor) Monitor.Log($"action[{i}", $"action[{i}: {jointAction} - decreaseSpeed by 1");
				cost = kAccelerateMod1 * 1.1f * 3f;
				cost += _previousActions[i] < 4 ? 1f : 0f;
				MuscleContiniousActions[i] -= kAccelerateMod1;
				break;
			case 5:
				if (printAction) print("decreaseSpeed by 2");
				if (ShowMonitor) Monitor.Log($"action[{i}", $"action[{i}: {jointAction} - decreaseSpeed by 2");
				cost = kAccelerateMod2 * 1.2f * 3f;
				cost += _previousActions[i] < 4 ? 1f : 0f;
				MuscleContiniousActions[i] -= kAccelerateMod2;
				break;
			case 6:
				if (printAction) print("decreaseSpeed by 3");
				if (ShowMonitor) Monitor.Log($"action[{i}", $"action[{i}: {jointAction} - decreaseSpeed by 3");
				cost = kAccelerateMod3 * 1.3f * 3f;
				cost += _previousActions[i] < 4 ? 1f : 0f;
				MuscleContiniousActions[i] -= kAccelerateMod3;
				break;
		}
		MuscleContiniousActions[i] = Mathf.Clamp(MuscleContiniousActions[i], -1f, 1f);
		if (muscle.ObsNormalizedAngleX >= .999f)
			SetActionMask(0, new int[]{4,5,6});
		else if (muscle.ObsNormalizedAngleX <= -.999f)
			SetActionMask(0, new int[]{1,2,3});

		i = 0;
		foreach (var m in Muscles)
		{
            m.TargetAngularVelocityX = MuscleContiniousActions[i++];
            m.UpdateMotor();
        }

		_previousActions = vectorAction.ToList();
		ProcessReward(cost);
		if (RewardCount == null || RewardCount.Count == 0)
			RewardCount = Enumerable.Range(0, brain.brainParameters.vectorActionSize[0]).Select(x => 0).ToList();
		RewardCount[jointAction]++;
		var softmax = Softmax(RewardCount.ToList());
		if (ShowMonitor) 
			Monitor.Log($"actions:", softmax);
			// Monitor.Log($"actions:", softmax, null, Monitor.DisplayType.PROPORTION);
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
            AddVectorObs(_posDistance);
            AddVectorObs(_negDistance);
            AddVectorObs(m.TargetAngularVelocityX);
            AddVectorObs(m.ObsNormalizedAngleX);
            AddVectorObs(m.TargetNormalizedAngleX);
            AddVectorObs(m.ObsRotationVelocity);
            AddVectorObs(_isStill);
            AddVectorObs(_isOnTarget);
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

    void ProcessReward(float cost)
	{
		bool recalcPenality = _penalty == null;
		if (recalcPenality)
			_penalty = 0f;		
		int i = 0;
		if (_previousMuscleContiniousActions == null)
			_previousMuscleContiniousActions = MuscleContiniousActions?.ToList();
		while (recalcPenality)
		{
			var difference = Mathf.Abs(_previousMuscleContiniousActions[i]-MuscleContiniousActions[i]);
			_penalty += difference;
			if (_previousMuscleContiniousActions[i] * MuscleContiniousActions[i] < 0.0f)
				_penalty += 1f; // signs are different
			i++;
			if (i>=MuscleContiniousActions.Count)
				break;
		}
		_previousMuscleContiniousActions = MuscleContiniousActions?.ToList();
		Rewards = new List<float>();
		foreach (var m in Muscles)
		{
			m.UpdateObservations();
			Rewards.Add(m.GetReward());
		}
		var hist = new List<float> ();

		var reward = Rewards
			.Average();
		hist.Add(reward);

		// reward = Mathf.Pow(reward, 2);
		// hist.Add(reward);
		// cost *= .5f;
		hist.Add(-cost);
		hist.Add(-_penalty.Value);
		hist.Add(_isStill ? 1f : 0f);
		hist.Add(_isOnTarget ? 1f : 0f);

		reward -= cost;
		if (_isStill && _isOnTarget)
			reward += .5f;
		if (_isOnTarget)
			reward += 1f- Mathf.Clamp(Mathf.Abs(Muscles[0].ObsRotationVelocity), 0f, 1f);
		hist.Add(reward);
		AddReward(reward);
		if (ShowMonitor) 
			Monitor.Log("rewardHist", hist.ToArray());
		_jointTrainerAgent.ChildStep(reward);		
    }

	float[] Softmax (List<int> input)
	{
		var inputAsDoubles = input.Select(x=>(double)x).ToArray();
		return Softmax(inputAsDoubles);
	}

	float[] Softmax(double[] oSums)
	{
		double sum = oSums.Sum(d => Math.Abs(d));

		double[] result = new double[oSums.Length];
		for (int i = 0; i < oSums.Length; ++i)
			result[i] = Math.Abs(oSums[i]) / sum;

		// scaled so that xi sum to 1.0
		return result.Select(x=>(float)x).ToArray();
	}
}